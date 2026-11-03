import json
from typing import Optional
from functools import cached_property
from tempfile import TemporaryDirectory
import arxiv
import tarfile
import gzip
import re
import time
from llm import get_llm, ClaudeCLI
import requests
from requests.adapters import HTTPAdapter, Retry
from loguru import logger
import tiktoken
from contextlib import ExitStack
from urllib.error import HTTPError



class ArxivPaper:
    # Titles from the reader's Zotero library, shown to the Claude backend for associations; set in main.py
    library_titles: list[str] = []

    def __init__(self,paper:arxiv.Result):
        self._paper = paper
        self.score = None
    
    @property
    def title(self) -> str:
        return self._paper.title
    
    @property
    def summary(self) -> str:
        return self._paper.summary
    
    @property
    def authors(self) -> list[str]:
        return self._paper.authors
    
    @cached_property
    def arxiv_id(self) -> str:
        return re.sub(r'v\d+$', '', self._paper.get_short_id())
    
    @property
    def pdf_url(self) -> str:
        return self._paper.pdf_url
    
    @cached_property
    def code_url(self) -> Optional[str]:
        s = requests.Session()
        retries = Retry(total=5, backoff_factor=0.1)
        s.mount('https://', HTTPAdapter(max_retries=retries))
        try:
            paper_list = s.get(f'https://paperswithcode.com/api/v1/papers/?arxiv_id={self.arxiv_id}').json()
        except Exception as e:
            logger.debug(f'Error when searching {self.arxiv_id}: {e}')
            return None

        if paper_list.get('count',0) == 0:
            return None
        paper_id = paper_list['results'][0]['id']

        try:
            repo_list = s.get(f'https://paperswithcode.com/api/v1/papers/{paper_id}/repositories/').json()
        except Exception as e:
            logger.debug(f'Error when searching {self.arxiv_id}: {e}')
            return None
        if repo_list.get('count',0) == 0:
            return None
        return repo_list['results'][0]['url']
    
    @cached_property
    def tex(self) -> dict[str,str]:
        with ExitStack() as stack:
            tmpdirname = stack.enter_context(TemporaryDirectory())
            # file = self._paper.download_source(dirpath=tmpdirname)
            try:
                # 尝试下载源文件
                file = self._paper.download_source(dirpath=tmpdirname)
            except HTTPError as e:
                # 捕获 HTTP 错误
                if e.code == 404:
                    # 如果是 404 Not Found，说明源文件不存在，这是正常情况
                    logger.warning(f"Source for {self.arxiv_id} not found (404). Skipping source analysis.")
                    return None # 直接返回 None，后续依赖 tex 的代码会安全地处理
                else:
                    # 如果是其他 HTTP 错误 (如 503)，这可能是临时性问题，值得记录下来
                    logger.error(f"HTTP Error {e.code} when downloading source for {self.arxiv_id}: {e.reason}")
                    raise # 重新抛出异常，因为这可能是个需要关注的严重问题
            try:
                tar = stack.enter_context(tarfile.open(file))
                names = tar.getnames()
                read = lambda name: tar.extractfile(name).read()
            except tarfile.ReadError:
                # single-file submissions come as one gzipped .tex instead of a tar
                try:
                    with gzip.open(file) as f:
                        single = f.read()
                except (OSError, EOFError):
                    single = b''
                # LaTeX, LaTeX 2.09 or plain TeX; anything else (e.g. PostScript) is not usable
                if not re.search(rb'\\documentclass|\\documentstyle|\\begin\{document\}|\\bye\b', single):
                    logger.debug(f"Failed to find main tex file of {self.arxiv_id}: Not a tar file or a gzipped tex file.")
                    return None
                names = ['main.tex']
                read = lambda name: single

            tex_files = [f for f in names if f.endswith('.tex')]
            if len(tex_files) == 0:
                logger.debug(f"Failed to find main tex file of {self.arxiv_id}: No tex file.")
                return None
            
            bbl_file = [f for f in names if f.endswith('.bbl')]
            match len(bbl_file) :
                case 0:
                    if len(tex_files) > 1:
                        logger.debug(f"Cannot find main tex file of {self.arxiv_id} from bbl: There are multiple tex files while no bbl file.")
                        main_tex = None
                    else:
                        main_tex = tex_files[0]
                case 1:
                    main_name = bbl_file[0].replace('.bbl','')
                    main_tex = f"{main_name}.tex"
                    if main_tex not in tex_files:
                        logger.debug(f"Cannot find main tex file of {self.arxiv_id} from bbl: The bbl file does not match any tex file.")
                        main_tex = None
                case _:
                    logger.debug(f"Cannot find main tex file of {self.arxiv_id} from bbl: There are multiple bbl files.")
                    main_tex = None
            if main_tex is None:
                logger.debug(f"Trying to choose tex file containing the document block as main tex file of {self.arxiv_id}")
            #read all tex files
            file_contents = {}
            for t in tex_files:
                content = read(t).decode('utf-8',errors='ignore')
                #remove comments (an escaped \% is a literal percent sign, not a comment)
                content = re.sub(r'(?<!\\)%.*\n', '\n', content)
                content = re.sub(r'\\begin{comment}.*?\\end{comment}', '', content, flags=re.DOTALL)
                content = re.sub(r'\\iffalse.*?\\fi', '', content, flags=re.DOTALL)
                #remove redundant \n
                content = re.sub(r'\n+', '\n', content)
                # keep \\ as a line break so table rows stay apart
                content = re.sub(r'\\\\', '\n', content)
                #remove consecutive spaces
                content = re.sub(r'[ \t\r\f]{3,}', ' ', content)
                if main_tex is None and re.search(r'\\begin\{document\}', content):
                    main_tex = t
                    logger.debug(f"Choose {t} as main tex file of {self.arxiv_id}")
                file_contents[t] = content
            
            if main_tex is not None:
                main_source:str = file_contents[main_tex]
                #find and replace all included sub-files
                include_files = re.findall(r'\\input\{(.+?)\}', main_source) + re.findall(r'\\include\{(.+?)\}', main_source)
                for f in include_files:
                    if not f.endswith('.tex'):
                        file_name = f + '.tex'
                    else:
                        file_name = f
                    main_source = main_source.replace(f'\\input{{{f}}}', file_contents.get(file_name, ''))
                file_contents["all"] = main_source
            else:
                logger.debug(f"Failed to find main tex file of {self.arxiv_id}: No tex file containing the document block.")
                file_contents["all"] = None
        return file_contents

    @cached_property
    def full_text(self) -> Optional[str]:
        """LaTeX body with figures, citations and layout commands removed, for the Claude backend. None if there is no tex source."""
        if self.tex is None:
            return None
        files = {k: v for k, v in self.tex.items() if k != "all" and v is not None}
        content = self.tex.get("all") or "\n".join(files.values())
        # "all" only expands \input one level deep; also resolve nested \input and \include
        def expand(m):
            name = m.group(1).strip().removeprefix('./')
            return files.get(name if name.endswith('.tex') else name + '.tex', '')
        for _ in range(5):
            content, n = re.subn(r'\\(?:input|include)\{(.+?)\}', expand, content)
            if n == 0:
                break
        # drop the preamble but keep its macro definitions, which the body relies on
        preamble, sep, body = content.partition('\\begin{document}')
        if sep:
            macros = re.findall(r'^\s*\\(?:newcommand|renewcommand|providecommand|def|DeclareMathOperator)\b.*$', preamble, flags=re.MULTILINE)
            content = '\n'.join(macros + [body.split('\\end{document}')[0]])
        content = re.sub(r'\\begin\{(figure\*?|wrapfigure|SCfigure|tikzpicture)\}.*?\\end\{\1\}', '', content, flags=re.DOTALL)
        content = re.sub(r'\\begin\{thebibliography\}.*?\\end\{thebibliography\}', '', content, flags=re.DOTALL)
        content = re.sub(r'\\includegraphics(\[[^\]]*\])?\{[^}]*\}', '', content)
        content = re.sub(r'~?\\cite[a-zA-Z]*\*?(\[[^\]]*\])*\{[^}]*\}', '', content)
        content = re.sub(r'\\(vspace|hspace|bibliographystyle|bibliography)\*?\{[^}]*\}', '', content)
        content = re.sub(r'\\(maketitle|centering|noindent|newpage|clearpage|cleardoublepage|smallskip|medskip|bigskip|vfill|hfill|FloatBarrier|printbibliography)\b', '', content)
        content = re.sub(r'\s*\n\s*', '\n', content).strip()
        if not content:
            return None
        # use gpt-4o tokenizer for estimation; cap far below the context window
        enc = tiktoken.encoding_for_model("gpt-4o")
        tokens = enc.encode(content)
        if len(tokens) > 100000:
            logger.debug(f"Truncating full text of {self.arxiv_id} from {len(tokens)} to 100000 tokens.")
            content = enc.decode(tokens[:100000])
        return content

    @cached_property
    def digest(self) -> dict[str, str]:
        """TLDR, an overview and associations for the Claude Code prompt, from one Claude call over the full text."""
        llm = get_llm()
        prompt = """Below is the LaTeX source of a paper, with figures, citations and layout commands removed. Return three fields:
- tldr: a one-sentence TLDR summary in __LANG__.
- overview: a briefing in __LANG__, under 300 words, for an assistant who will discuss this paper with a reader but has not read it. Cover the problem, the core method (with the key equations if they are central), the main results with numbers, the limitations, and points worth questioning.
- associations: in __LANG__, under 250 words, in two parts. First, which papers in the reader's library (titles listed in the system prompt) this paper connects to and how: building on them, competing with them, or combinable with them. Quote those titles exactly and name only genuinely related ones; say so if none are. Second, open-ended associations: related lines of research, possible extensions, uses in other fields, and ideas the reader could try. Describe directions rather than citing specific papers unless you are sure they exist.

\\title{__TITLE__}
\\begin{abstract}__ABSTRACT__\\end{abstract}
__FULL_TEXT__
"""
        prompt = prompt.replace('__LANG__', llm.lang)
        prompt = prompt.replace('__TITLE__', self.title)
        prompt = prompt.replace('__ABSTRACT__', self.summary)
        prompt = prompt.replace('__FULL_TEXT__', self.full_text)
        # the library goes in the system prompt: it is identical for every paper, so later calls hit the prompt cache
        system = "You are an assistant who perfectly summarizes scientific paper, and gives the core idea of the paper to the user."
        system += "\n\nTitles of the papers in the reader's library:\n" + "\n".join(f"- {t}" for t in self.library_titles)
        return llm.generate(
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
            json_schema={
                "type": "object",
                "properties": {"tldr": {"type": "string"}, "overview": {"type": "string"}, "associations": {"type": "string"}},
                "required": ["tldr", "overview", "associations"],
            },
        )

    @cached_property
    def cc_prompt(self) -> Optional[str]:
        """A prompt to paste into Claude Code to discuss this paper. Only built with the Claude backend."""
        llm = get_llm()
        if not isinstance(llm, ClaudeCLI):
            return None
        return f"""I'd like to discuss the arXiv paper "{self.title}" with you.

arXiv: https://arxiv.org/abs/{self.arxiv_id}
DOI: 10.48550/arXiv.{self.arxiv_id}
PDF: {self.pdf_url}

Overview, written by Claude after reading the paper's LaTeX source (it may contain mistakes):
{self.digest['overview']}

Associations, also by Claude: links to papers in my library and open-ended ideas (leads to check, not facts):
{self.digest['associations']}

Instructions:
- Read the paper itself (HTML version: https://arxiv.org/html/{self.arxiv_id}, or the PDF) before answering detailed questions; don't rely on the overview alone.
- Say which section, equation or table each answer relies on, and keep what the paper states apart from your own inference.
- Answer in {llm.lang}. Confirm once you have read the paper, then wait for my questions."""

    @cached_property
    def tldr(self) -> str:
        if isinstance(get_llm(), ClaudeCLI):
            return self.digest['tldr']
        introduction = ""
        conclusion = ""
        if self.tex is not None:
            content = self.tex.get("all")
            if content is None:
                content = "\n".join(v for v in self.tex.values() if v is not None)
            #remove cite
            content = re.sub(r'~?\\cite.?\{.*?\}', '', content)
            #remove figure
            content = re.sub(r'\\begin\{figure\}.*?\\end\{figure\}', '', content, flags=re.DOTALL)
            #remove table
            content = re.sub(r'\\begin\{table\}.*?\\end\{table\}', '', content, flags=re.DOTALL)
            #find introduction and conclusion
            # end word can be \section or \end{document} or \bibliography or \appendix
            match = re.search(r'\\section\{Introduction\}.*?(\\section|\\end\{document\}|\\bibliography|\\appendix|$)', content, flags=re.DOTALL)
            if match:
                introduction = match.group(0)
            match = re.search(r'\\section\{Conclusion\}.*?(\\section|\\end\{document\}|\\bibliography|\\appendix|$)', content, flags=re.DOTALL)
            if match:
                conclusion = match.group(0)
        llm = get_llm()
        prompt = """Given the title, abstract, introduction and the conclusion (if any) of a paper in latex format, generate a one-sentence TLDR summary in __LANG__:
        
        \\title{__TITLE__}
        \\begin{abstract}__ABSTRACT__\\end{abstract}
        __INTRODUCTION__
        __CONCLUSION__
        """
        prompt = prompt.replace('__LANG__', llm.lang)
        prompt = prompt.replace('__TITLE__', self.title)
        prompt = prompt.replace('__ABSTRACT__', self.summary)
        prompt = prompt.replace('__INTRODUCTION__', introduction)
        prompt = prompt.replace('__CONCLUSION__', conclusion)

        # use gpt-4o tokenizer for estimation
        enc = tiktoken.encoding_for_model("gpt-4o")
        prompt_tokens = enc.encode(prompt)
        prompt_tokens = prompt_tokens[:4000]  # truncate to 4000 tokens
        prompt = enc.decode(prompt_tokens)
        
        tldr = llm.generate(
            messages=[
                {
                    "role": "system",
                    "content": "You are an assistant who perfectly summarizes scientific paper, and gives the core idea of the paper to the user.",
                },
                {"role": "user", "content": prompt},
            ]
        )
        return tldr
    
    @cached_property
    def key_concepts(self) -> dict[str,str]:
        introduction = ""
        conclusion = ""
        if self.tex is not None:
            content = self.tex.get("all")
            if content is None:
                content = "\n".join(v for v in self.tex.values() if v is not None)
            #remove cite
            content = re.sub(r'~?\\cite.?\{.*?\}', '', content)
            #remove figure
            content = re.sub(r'\\begin\{figure\}.*?\\end\{figure\}', '', content, flags=re.DOTALL)
            #remove table
            content = re.sub(r'\\begin\{table\}.*?\\end\{table\}', '', content, flags=re.DOTALL)
            #find introduction and conclusion
            # end word can be \section or \end{document} or \bibliography or \appendix
            match = re.search(r'\\section\{Introduction\}.*?(\\section|\\end\{document\}|\\bibliography|\\appendix|$)', content, flags=re.DOTALL)
            if match:
                introduction = match.group(0)
            match = re.search(r'\\section\{Conclusion\}.*?(\\section|\\end\{document\}|\\bibliography|\\appendix|$)', content, flags=re.DOTALL)
            if match:
                conclusion = match.group(0)
        llm = get_llm()
        prompt = """Given the title, abstract, introduction, conclusion and a brief TLDR of a paper in latex format, give the explanation of some key concepts, especially the concepts in the TLDR.
        
        The key concepts should be in both __LANG__ and the original language of the paper, and the explanations should be in __LANG__ only.

        Your response should be a JSON object, where the keys are the key concepts in both languages and the values are the explanations of the key concepts. You should only return the JSON object, WITHOUT any other text like ``` or ```json.

        \\title{__TITLE__}
        \\begin{abstract}__ABSTRACT__\\end{abstract}
        \\begin{tldr}__TLDR__\\end{tldr}
        __INTRODUCTION__
        __CONCLUSION__
        """
        prompt = prompt.replace('__LANG__', llm.lang)
        prompt = prompt.replace('__TITLE__', self.title)
        prompt = prompt.replace('__ABSTRACT__', self.summary)
        # the TLDR sits before the introduction so the 4000-token truncation below cannot cut it
        prompt = prompt.replace('__TLDR__', self.tldr)
        prompt = prompt.replace('__INTRODUCTION__', introduction)
        prompt = prompt.replace('__CONCLUSION__', conclusion)

        # use gpt-4o tokenizer for estimation
        enc = tiktoken.encoding_for_model("gpt-4o")
        prompt_tokens = enc.encode(prompt)
        prompt_tokens = prompt_tokens[:4000]  # truncate to 4000 tokens
        prompt = enc.decode(prompt_tokens)

        key_concepts = llm.generate(
            messages=[
                {
                    "role": "system",
                    "content": "You are an assistant who perfectly summarizes scientific paper, and gives the core idea of the paper to the user.",
                },
                {"role": "user", "content": prompt},
            ]
        )
        try:
            # take the outermost {...} so stray code fences or a leading sentence don't break parsing
            key_concepts = json.loads(re.search(r'\{.*\}', key_concepts, flags=re.DOTALL).group(0))
        except Exception as e:
            logger.debug(f"Failed to extract key concepts of {self.arxiv_id}: {e}")
            key_concepts = {}
        if not isinstance(key_concepts, dict):
            logger.debug(f"Failed to extract key concepts of {self.arxiv_id}: Not a dictionary.")
            return {}
        return {str(k):str(v) for k,v in key_concepts.items()}

    @cached_property
    def affiliations(self) -> Optional[list[str]]:
        if self.tex is not None:
            content = self.tex.get("all")
            if content is None:
                content = "\n".join(v for v in self.tex.values() if v is not None)
            #search for affiliations
            possible_regions = [r'\\author.*?\\maketitle',r'\\begin{document}.*?\\begin{abstract}']
            matches = [re.search(p, content, flags=re.DOTALL) for p in possible_regions]
            match = next((m for m in matches if m), None)
            if match:
                information_region = match.group(0)
            else:
                logger.debug(f"Failed to extract affiliations of {self.arxiv_id}: No author information found.")
                return None
            prompt = f"Given the author information of a paper in latex format, extract the affiliations of the authors in a python list format, which is sorted by the author order. If there is no affiliation found, return an empty list '[]'. Following is the author information:\n{information_region}"
            # use gpt-4o tokenizer for estimation
            enc = tiktoken.encoding_for_model("gpt-4o")
            prompt_tokens = enc.encode(prompt)
            prompt_tokens = prompt_tokens[:4000]  # truncate to 4000 tokens
            prompt = enc.decode(prompt_tokens)
            llm = get_llm()
            affiliations = llm.generate(
                messages=[
                    {
                        "role": "system",
                        "content": "You are an assistant who perfectly extracts affiliations of authors from the author information of a paper. You should return a python list of affiliations sorted by the author order, like ['TsingHua University','Peking University']. If an affiliation is consisted of multi-level affiliations, like 'Department of Computer Science, TsingHua University', you should return the top-level affiliation 'TsingHua University' only. Do not contain duplicated affiliations. If there is no affiliation found, you should return an empty list [ ]. You should only return the final list of affiliations, and do not return any intermediate results.",
                    },
                    {"role": "user", "content": prompt},
                ]
            )

            try:
                affiliations = re.search(r'\[.*?\]', affiliations, flags=re.DOTALL).group(0)
                affiliations = eval(affiliations)
                affiliations = list(set(affiliations))
                affiliations = [str(a) for a in affiliations]
            except Exception as e:
                logger.debug(f"Failed to extract affiliations of {self.arxiv_id}: {e}")
                return None
            return affiliations
