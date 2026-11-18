"""Export the recorded course report as editable LaTeX; no model computation.

The generated report.tex is self-contained apart from copied figures/files.
It can be edited directly and compiled with XeLaTeX or Tectonic.
"""
from __future__ import annotations

import argparse
from datetime import date, datetime
import hashlib
import json
from pathlib import Path
import re
import shutil
import shlex
from zoneinfo import ZoneInfo

from markdown_it import MarkdownIt

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'reports/course_draft_20261115'
DEST = SOURCE / 'latex'
MD = MarkdownIt('commonmark').enable('table')


def project_path(path: Path) -> str:
    """Prefer project-relative provenance, while allowing explicit external inputs."""
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path.resolve())

MATH = {
    'Σr_t': r'\sum_{t=0}^{T-1} r_t',
    'Return = Σr_t': r'R=\sum_{t=0}^{T-1}r_t',
    'Return/30': r'R/30',
    '√mean((v_x−0.10)²)': r'\sqrt{\operatorname{mean}_t[(v_{x,t}-0.10)^2]}',
    '√mean(wrap(ψ−ψ_cmd)²)': r'\sqrt{\operatorname{mean}_t[\operatorname{wrap}(\psi_t-\psi_{\mathrm{cmd}})^2]}',
    '√mean_t((vx−vx_ref)²+(vy−vy_ref)²)': r'\sqrt{\operatorname{mean}_t[(v_{x,t}-v^{\mathrm{ref}}_{x,t})^2+(v_{y,t}-v^{\mathrm{ref}}_{y,t})^2]}',
    '√mean_t(wrap(ψ−ψ_ref)²)': r'\sqrt{\operatorname{mean}_t[\operatorname{wrap}(\psi_t-\psi_t^{\mathrm{ref}})^2]}',
    '√mean_t((x−x_ref)²+(y−y_ref)²+(z−z_ref)²)': r'\sqrt{\operatorname{mean}_t[(x_t-x_t^{\mathrm{ref}})^2+(y_t-y_t^{\mathrm{ref}})^2+(z_t-z_t^{\mathrm{ref}})^2]}',
    'v': r'\mathbf v', 'ω': r'\boldsymbol\omega', 'u': 'u',
    'a': r'\mathbf a', 'mean': r'\operatorname{mean}_j', '[0,1]': '[0,1]',
    'ψ_cmd=0': r'\psi_{\mathrm{cmd}}=0',
    '1.5 exp(−((vx−0.1)²+vy²)/0.01)u²': r'1.5\exp\!\left[-\frac{(v_x-0.1)^2+v_y^2}{0.01}\right]u^2',
    'clip(vx/0.1,−1,1)u²': r'\operatorname{clip}(v_x/0.1,-1,1)u^2',
    '−0.5(1−cos(ψ−ψ_cmd))': r'-0.5[1-\cos(\psi-\psi_{\mathrm{cmd}})]',
    '+0.35u': r'+0.35u',
    '−0.10(vy/0.10)²': r'-0.10(v_y/0.10)^2',
    '−0.03ωz²': r'-0.03\omega_z^2',
    '−0.01(ωx²+ωy²)': r'-0.01(\omega_x^2+\omega_y^2)',
    '−0.02(vz/0.10)²': r'-0.02(v_z/0.10)^2',
    '−0.02 mean((τ/0.18)²)': r'-0.02\operatorname{mean}_j[(\tau_j/0.18)^2]',
    '−0.08 mean((a_t−a_(t−1))²)': r'-0.08\operatorname{mean}_j[(a_{t,j}-a_{t-1,j})^2]',
    '−0.10 mean((q−q_stand)²)': r'-0.10\operatorname{mean}_j[(q_j-q^{\mathrm{stand}}_j)^2]',
    '−2vz²': r'-2v_z^2', 'vx−0.10': r'v_x-0.10',
    'w × exp(−误差平方/尺度²)': r'w\exp(-e^2/\sigma^2)',
    'r_t = 0.02 × 奖励率总和 − 2 × I(真正终止)': r'r_t=0.02\sum_k\rho_{k,t}-2\,\mathbf{1}_{\{\text{真正终止}\},t}',
}

SPECIAL = {
    '\\': r'\textbackslash{}', '{': r'\{', '}': r'\}', '&': r'\&',
    '%': r'\%', '$': r'\$', '#': r'\#', '_': r'\_', '~': r'\textasciitilde{}',
    '^': r'\textasciicircum{}', '±': r'\(\pm\)', '×': r'\(\times\)',
    '−': r'\(-\)', '²': r'\textsuperscript{2}', '°': r'\(^{\circ}\)',
    'λ': r'\(\lambda\)', 'μ': r'\(\mu\)', '→': r'\(\rightarrow\)',
}

TEXT_MATH = {
    'Δq̇ / 0.02 秒': r'\(\Delta\dot q/\Delta t\)（\(\Delta t=0.02\,\mathrm{s}\)）',
    'Δq̇ / 0.02': r'\(\Delta\dot q/0.02\)',
    'Δt': r'\(\Delta t\)',
}
TEXT_PATTERN = re.compile('|'.join(re.escape(k) for k in TEXT_MATH))


def escape(value: str) -> str:
    parts, previous = [], 0
    for match in TEXT_PATTERN.finditer(value):
        parts.append(''.join(SPECIAL.get(c, c) for c in value[previous:match.start()]))
        parts.append(TEXT_MATH[match.group()]); previous = match.end()
    parts.append(''.join(SPECIAL.get(c, c) for c in value[previous:]))
    return ''.join(parts)


def inline(source: str) -> str:
    # Move Chinese punctuation outside emphasis for CommonMark delimiter rules.
    source = source.replace('**评测范围：**', '**评测范围**：')
    out = []
    for token in MD.parseInline(source)[0].children:
        if token.type == 'text': out.append(escape(token.content))
        elif token.type == 'code_inline':
            if token.content in MATH:
                out.append(r'\('+MATH[token.content]+r'\)')
            else:
                if any(c in token.content for c in '{}%'):
                    raise ValueError(f'Unexpected verbatim argument: {token.content}')
                # url/nolinkurl discard spaces; keep command separators explicit.
                parts = token.content.split(' ')
                out.append(r'\ '.join(r'\nolinkurl{'+part+'}' for part in parts))
        elif token.type == 'strong_open': out.append(r'\textbf{')
        elif token.type == 'strong_close': out.append('}')
        elif token.type == 'em_open': out.append(r'\emph{')
        elif token.type == 'em_close': out.append('}')
        elif token.type == 'link_open':
            url = token.attrGet('href')
            if not url.startswith(('https://', 'http://')):
                url = 'run:'+copy_link(url)
            out.append(r'\href{'+url.replace('%', r'\%').replace('#', r'\#')+'}{')
        elif token.type == 'link_close': out.append('}')
        elif token.type in ('softbreak', 'hardbreak'): out.append(' ')
        else: raise ValueError(f'Unexpected inline token: {token.type}')
    return ''.join(out)


COPIED = {}
PAPER_STYLE = None


def style_asset(relative: str) -> Path:
    """Resolve a report-local override without permitting path traversal."""
    source = (SOURCE/relative).resolve()
    if Path(relative).is_absolute() or not source.is_relative_to(SOURCE.resolve()):
        raise ValueError(f'Paper-style asset must be relative to SOURCE: {relative}')
    if not source.is_file():
        raise FileNotFoundError(source)
    return source


def load_paper_style() -> dict:
    source = SOURCE/'paper_style/report_overrides.json'
    data = json.loads(source.read_text())
    if not isinstance(data, dict):
        raise ValueError('Paper-style overrides must be a JSON object')
    for name in ('figure_sources', 'caption_overrides', 'link_sources'):
        mapping = data.get(name, {})
        if not isinstance(mapping, dict) or not all(
                isinstance(k, str) and isinstance(v, str) for k, v in mapping.items()):
            raise ValueError(f'Paper-style {name} must map strings to strings')
        data[name] = mapping
    assets = data.get('extra_assets', [])
    if not isinstance(assets, list) or not all(isinstance(p, str) for p in assets):
        raise ValueError('Paper-style extra_assets must be a list of relative paths')
    data['extra_assets'] = assets
    if not isinstance(data.get('note'), str) or not data['note'].strip():
        raise ValueError('Paper-style note must describe the plotting/smoothing changes')
    for number, caption in data['caption_overrides'].items():
        if not re.fullmatch(r'[1-9]\d*', number) or re.match(r'^图\s*\d+\s*[：:]', caption):
            raise ValueError(f'Invalid paper-style caption override: {number}')
    for relative in [*data['figure_sources'].values(), *data['link_sources'].values(), *assets]:
        style_asset(relative)
    return data


def copy_file(source: Path, relative: str) -> str:
    target = DEST/relative
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)
    COPIED[relative] = dict(source=project_path(source),
                            sha256=hashlib.sha256(source.read_bytes()).hexdigest())
    return relative


def copy_link(relative: str) -> str:
    source = (SOURCE/relative).resolve()
    if source.is_relative_to(SOURCE):
        target = source.relative_to(SOURCE).as_posix()
    else:
        target = 'support/'+source.name
    if PAPER_STYLE is not None and relative in PAPER_STYLE['link_sources']:
        source = style_asset(PAPER_STYLE['link_sources'][relative])
    if not source.is_file(): raise FileNotFoundError(source)
    return copy_file(source, target)


TABLE_NAMES = [
    '仿真环境与任务设置', '三种基线的共同测试结果', '基线训练时间',
    'Isaac 行走任务的奖励率及作用', '平滑性改进的共同测试结果',
    '0.20 m/s 指令下的共同测试结果', '动作平滑性与高速行走实验的训练时间',
    '参考轨迹跟踪对照组', '参考轨迹跟踪奖励',
    '四组固定预算模型的最终测试', '最终测试回报按参考来源分组',
    '固定交互预算下的运动指标', '想象长度与 Critic 消融的最终测试',
    '参考跟踪实验的计算成本', '参考数据准备成本',
    '延长 PPO 预算后的相近时间窗口对照',
]

WIDTHS = {
    1: [.24,.76], 2:[.14,.10,.23,.30,.23], 3:[.22,.39,.39],
    4: [.14,.49,.37], 5:[.08,.15,.10,.12,.13,.10,.11,.10,.11],
    6: [.08,.16,.10,.11,.11,.10,.10,.11,.13],
    7:[.28,.32,.40], 8:[.23,.24,.27,.26], 9:[.27,.13,.20,.40],
    10:[.30,.17,.19,.17,.17], 11:[.30,.23,.235,.235],
    12:[.28,.16,.15,.16,.15,.10], 13:[.30,.17,.19,.17,.17],
    14:[.27,.115,.135,.14,.14,.09,.11], 15:[.28,.14,.12,.17,.12,.17],
    16:[.30,.25,.24,.21],
}


def table_layout_indices(metadata: dict) -> list[int]:
    """Map consecutive report tables to the original named layout definitions."""
    indices = metadata.get('table_layout_source_indices', list(range(1, len(TABLE_NAMES) + 1)))
    if (not isinstance(indices, list) or not indices or
            any(type(index) is not int or index not in WIDTHS for index in indices) or
            len(indices) != len(set(indices))):
        raise ValueError('table_layout_source_indices must contain distinct valid table indices')
    expected = metadata.get('table_count', len(indices))
    if type(expected) is not int or expected != len(indices):
        raise ValueError('Table layout mapping differs from report table_count')
    return indices


def table(rows: list[list[str]], number: int, layout_number: int | None = None) -> str:
    layout_number = number if layout_number is None else layout_number
    n = len(rows[0]); assert all(len(row) == n for row in rows)
    matching_source_table = layout_number == 11 and n == 3
    proportions = [.43, .30, .27] if matching_source_table else WIDTHS[layout_number]
    if layout_number == 15 and n == 5:
        proportions = [.34, .18, .14, .20, .14]
    total = sum(proportions)
    if n != len(proportions):
        raise ValueError(f'Table {number} has {n} columns; layout {layout_number} expects {len(proportions)}')
    # Weighted X columns sum to the original number of X columns.
    columns = []
    for j, weight in enumerate(proportions):
        align = r'\raggedright' if j == 0 or layout_number in (1,4,8,9,15) else r'\centering'
        columns.append(r'>{\hsize='+f'{n*weight/total:.6f}'+r'\hsize\linewidth=\hsize'+align+r'\arraybackslash}X')
    font = r'\fontsize{7.6}{10.2}\selectfont' if n >= 8 else r'\fontsize{8.4}{11.2}\selectfont'
    caption = ('各方法在对应来源参考轨迹上的测试回报' if matching_source_table
               else TABLE_NAMES[layout_number-1])
    body = [r'\begin{table}[!htbp]', r'\centering', r'\caption{'+inline(caption)+'}',
            r'\label{tab:results-'+str(number)+'}', r'\begingroup', font,
            r'\setlength{\tabcolsep}{3pt}', r'\renewcommand{\arraystretch}{1.20}',
            r'\begin{tabularx}{\linewidth}{@{}'+''.join(columns)+r'@{}}', r'\toprule']
    for i, row in enumerate(rows):
        cells = [inline(cell) for cell in row]
        if i == 0: cells = [r'\textbf{'+cell+'}' for cell in cells]
        body.append(' & '.join(cells)+r' \\')
        if i == 0: body.append(r'\midrule')
    body += [r'\bottomrule',r'\end{tabularx}',r'\endgroup',r'\end{table}',r'\FloatBarrier']
    return '\n'.join(body)


PREAMBLE = r'''% !TeX program = xelatex
% Editable course report. Compile report.tex with XeLaTeX (twice) or Tectonic.
% Experiments: 2026-11-15. LaTeX edition: 2026-11-16.
\documentclass[UTF8,a4paper,zihao=5,fontset=fandol]{ctexart}
\usepackage[left=18mm,right=18mm,top=20mm,bottom=21mm]{geometry}
\usepackage{amsmath,amssymb,booktabs,tabularx,array,graphicx}
\usepackage{xcolor,caption,placeins,fancyhdr}
\usepackage[unicode,colorlinks=true,linkcolor=blue!45!black,urlcolor=blue!55!black]{hyperref}
\usepackage{xurl}
\hypersetup{pdftitle={Sesame 四足机器人行走控制的强化学习实验研究},pdfauthor={上海交通大学强化学习课程作业}}
\urlstyle{tt}
\ctexset{section={number=\arabic{section},format=\large\bfseries},
 subsection={number=\arabic{section}.\arabic{subsection},format=\normalsize\bfseries}}
\linespread{1.05}
\setlength{\parskip}{0.35em}
\setlength{\emergencystretch}{2em}
\setlength{\textfloatsep}{10pt plus 2pt minus 2pt}
\setlength{\intextsep}{10pt plus 2pt minus 2pt}
\captionsetup{font=small,labelfont=bf,labelsep=quad}
\pagestyle{fancy}
\fancyhf{}
\fancyhead[L]{\small Sesame 四足机器人行走实验}
\fancyhead[R]{\small 强化学习课程作业}
\fancyfoot[C]{\small\thepage}
\setlength{\headheight}{15pt}
\renewcommand{\headrulewidth}{0.3pt}
\begin{document}
\begin{center}
{\LARGE\bfseries Sesame 四足机器人行走控制的强化学习实验研究}\par
\vspace{0.5em}
上海交通大学强化学习课程作业\par
{\small 2026年11月}
\end{center}
\vspace{0.4em}
'''


def render_paragraph(source: str) -> str:
    if source.startswith(('本报告使用的直观指标为：', '评价指标定义为：')):
        remainder = source.split('。',1)[1]
        return ('评价指标包括回报、前向速度误差和航向误差：\n'
                r'\begin{align}'+'\n'
                r' R &= \sum_{t=0}^{T-1}r_t,\\'+'\n'
                r' E_{v_x} &= '+MATH['√mean((v_x−0.10)²)']+r',\\'+'\n'
                r' E_{\psi} &= '+MATH['√mean(wrap(ψ−ψ_cmd)²)']+'.\n'
                r'\end{align}'+'\n'+inline(remainder))
    if source.startswith('单步奖励为 '):
        remainder = source.split('。',1)[1]
        return ('设表中第 '+r'\(k\)'+' 项奖励率为 '+r'\(\rho_{k,t}\)'+'，则单步奖励为\n'
                r'\begin{equation}'+'\n'+MATH['r_t = 0.02 × 奖励率总和 − 2 × I(真正终止)']+r'.'+'\n'
                r'\end{equation}'+'\n'+inline(remainder))
    if source.startswith('本节误差均相对参考轨迹'):
        start = '先分别计算' if '先分别计算' in source else '先计算'
        remainder = source[source.index(start):]
        return ('参考轨迹跟踪误差包括平面速度 RMSE、偏航 RMS 与机身位置 RMSE：\n'
                r'\begin{align}'+'\n'
                r' E_{v,xy} &= '+MATH['√mean_t((vx−vx_ref)²+(vy−vy_ref)²)']+r',\\'+'\n'
                r' E_{\psi,\mathrm{ref}} &= '+MATH['√mean_t(wrap(ψ−ψ_ref)²)']+r',\\'+'\n'
                r' E_{p} &= '+MATH['√mean_t((x−x_ref)²+(y−y_ref)²+(z−z_ref)²)']+'.\n'
                r'\end{align}'+'\n'+'偏航误差换算成度后报告。'+inline(remainder))
    return inline(source)


def main():
    global PAPER_STYLE, SOURCE, DEST
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=SOURCE,
                        help='Report directory containing report.md and its assets')
    parser.add_argument('--output', type=Path,
                        help='LaTeX destination directory (default: SOURCE/latex)')
    parser.add_argument('--paper-style', action='store_true',
                        help='Use paper_style/report_overrides.json for figures, captions and linked assets')
    args = parser.parse_args()
    SOURCE = args.source.resolve()
    DEST = args.output.resolve() if args.output is not None else SOURCE/'latex'
    if DEST == SOURCE:
        raise ValueError('LaTeX output must differ from the source report directory')
    metadata_path = SOURCE/'report_manifest.json'
    metadata = json.loads(metadata_path.read_text()) if metadata_path.exists() else {}
    layout_indices = table_layout_indices(metadata)
    report_date = date.fromisoformat(metadata.get('date', '2026-11-15'))
    export_date = datetime.now(ZoneInfo('Asia/Shanghai')).date()
    PAPER_STYLE = load_paper_style() if args.paper_style else None
    COPIED.clear()
    DEST.mkdir(parents=True, exist_ok=True)
    content = (SOURCE/'report.md').read_text()
    tokens = MD.parse(content)
    preamble = PREAMBLE.replace('% Experiments: 2026-11-15. LaTeX edition: 2026-11-16.',
                              f'% Experiments: {report_date}. LaTeX edition: {export_date}.')
    preamble = preamble.replace('2026年11月', f'{report_date.year}年{report_date.month}月')
    tex = [preamble]
    tables, figures, headings = [], [], []
    i = 0
    while i < len(tokens):
        t = tokens[i]
        if t.type == 'heading_open':
            level = int(t.tag[1]); title = tokens[i+1].content
            if level == 1:
                i += 3; continue
            tex.append(r'\FloatBarrier')
            if title.startswith('附录'):
                tex.append(r'\section*{'+inline(title)+'}')
            else:
                title = re.sub(r'^\d+(?:\.\d+)*\.?\s+', '', title)
                if title in ('训练曲线与共同测试结果', '主对照结果'):
                    # Keep this heading with the large learning-curve figure.
                    tex.append(r'\clearpage')
                command = 'section' if level == 2 else 'subsection'
                tex.append('\\'+command+'{'+inline(title)+'}')
            headings.append(title); i += 3
        elif t.type == 'paragraph_open':
            p = tokens[i+1]
            if p.content.startswith('上海交通大学强化学习课程作业 ·'):
                i += 3; continue
            if p.children and p.children[0].type == 'image':
                image = p.children[0]; relative = image.attrGet('src')
                source = SOURCE/relative
                target_relative = source.relative_to(SOURCE).as_posix()
                if PAPER_STYLE is not None and relative in PAPER_STYLE['figure_sources']:
                    source = style_asset(PAPER_STYLE['figure_sources'][relative])
                # Keep the original high-resolution PNG. Some existing CJK plot
                # PDFs have font mappings that xdvipdfmx corrupts when embedding.
                target = copy_file(source, target_relative)
                assert tokens[i+3].type == 'paragraph_open'
                caption = tokens[i+4].content
                matched = re.fullmatch(r'图 (\d+)：(.*)',caption)
                assert matched and int(matched[1]) == len(figures)+1
                number = int(matched[1]); caption = matched[2]
                if PAPER_STYLE is not None:
                    caption = PAPER_STYLE['caption_overrides'].get(str(number), caption)
                figures.append(dict(number=number, file=target, caption=caption))
                height = ('0.54' if target in ('improvements/smoothness_comparison.png',
                           'supermimic_comparison.png', 'supermimic_ablation.png') else '0.40')
                tex += [r'\begin{figure}[!htbp]',r'\centering',
                        r'\includegraphics[width=\linewidth,height='+height+r'\textheight,keepaspectratio]{'+target+'}',
                        r'\caption{'+inline(caption)+'}',r'\label{fig:experiment-'+str(number)+'}',
                        r'\end{figure}',r'\FloatBarrier']
                i += 6
            else:
                tex.append(render_paragraph(p.content)); i += 3
        elif t.type == 'table_open':
            j = i+1; rows = []; row = None
            while tokens[j].type != 'table_close':
                if tokens[j].type == 'tr_open': row = []
                elif tokens[j].type == 'inline': row.append(tokens[j].content)
                elif tokens[j].type == 'tr_close': rows.append(row)
                j += 1
            tables.append(rows)
            if len(tables) > len(layout_indices):
                raise ValueError('Report contains more tables than its layout mapping')
            tex.append(table(rows, len(tables), layout_indices[len(tables)-1])); i=j+1
        else:
            raise ValueError(f'Unexpected block {t.type} at {i}')
        tex.append('')
    if len(tables) != len(layout_indices):
        raise ValueError('Report table count differs from its layout mapping')
    if 'figure_count' in metadata and len(figures) != metadata['figure_count']:
        raise ValueError('Report figure count differs from report_manifest.json')
    tex += [r'\end{document}', '']
    result = '\n'.join(tex)
    (DEST/'report.tex').write_text(result)
    # Include scalar data already referenced by the report, not model weights.
    for folder in ('', 'baselines', 'improvements'):
        for p in (SOURCE/folder).glob('*.csv'):
            copy_link(p.relative_to(SOURCE).as_posix())
    copy_link('baselines/provenance.json')
    if PAPER_STYLE is None:
        copy_file(SOURCE/'report.md','source_report.md')
    else:
        known_figures = {str(figure['number']) for figure in figures}
        unknown = set(PAPER_STYLE['caption_overrides']) - known_figures
        if unknown:
            raise ValueError(f'Caption overrides refer to missing figures: {sorted(unknown)}')
        derived_content = re.sub(
            r'^图 (\d+)：(.*)$',
            lambda match: '图 '+match[1]+'：'+PAPER_STYLE['caption_overrides'].get(match[1], match[2]),
            content, flags=re.MULTILINE)
        (DEST/'source_report.md').write_text(derived_content)
        COPIED['source_report.md'] = dict(
            source=project_path(SOURCE/'report.md'),
            source_sha256=hashlib.sha256(content.encode()).hexdigest(),
            sha256=hashlib.sha256(derived_content.encode()).hexdigest(),
            transformation='Paper-style figure captions replaced; original report.md unchanged.')
        copy_file(SOURCE/'paper_style/report_overrides.json', 'paper_style/report_overrides.json')
        for relative in PAPER_STYLE['extra_assets']:
            copy_file(style_asset(relative), relative)
    manifest = dict(source=project_path(SOURCE/'report.md'),
                    source_sha256=hashlib.sha256(content.encode()).hexdigest(),
                    tex_sha256=hashlib.sha256(result.encode()).hexdigest(),
                    export_date=str(export_date), experiment_report_date=str(report_date),
                    table_count=len(tables),figure_count=len(figures),headings=headings,
                    table_layout_source_indices=layout_indices,
                    tables=tables,figures=figures,assets=COPIED,
                    note='Recorded results converted to LaTeX; no experiments, resampling or curve smoothing.')
    if PAPER_STYLE is not None:
        manifest['paper_style'] = dict(
            overrides_source=project_path(SOURCE/'paper_style/report_overrides.json'),
            overrides_sha256=hashlib.sha256((SOURCE/'paper_style/report_overrides.json').read_bytes()).hexdigest(),
            figure_sources=PAPER_STYLE['figure_sources'],
            caption_overrides=PAPER_STYLE['caption_overrides'],
            link_sources=PAPER_STYLE['link_sources'], extra_assets=PAPER_STYLE['extra_assets'],
            plotting_and_smoothing_note=PAPER_STYLE['note'])
        manifest['note'] = ('Recorded results converted to LaTeX with paper-style plotting overrides. '
                            'Exporter performs no experiments, resampling or smoothing. '+PAPER_STYLE['note'])
    (DEST/'content_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
    export_command = ['python3', 'scripts/export_course_report_latex.py',
                      '--source', project_path(SOURCE), '--output', project_path(DEST)]
    if PAPER_STYLE is not None:
        export_command.append('--paper-style')
    compile_command = shlex.join(['.runtime/latex_toolchain/bin/tectonic-local', '--untrusted',
                                 '--outdir', project_path(DEST), project_path(DEST/'report.tex')])
    instructions = '''# LaTeX 报告编译说明

主文件为 `report.tex`，可直接编辑。正文包含 __TABLE_COUNT__ 张结果/配置表和 __FIGURE_COUNT__ 张图，数学公式使用 LaTeX 数学环境。实验报告日期为 __REPORT_DATE__，LaTeX 导出日期为 __EXPORT_DATE__。

插图使用原报告的高分辨率 PNG，避免部分矢量 PDF 在嵌入时发生中文字体映射错误；曲线与实测数据未改动。

## Overleaf

上传整个源码 ZIP，主文件选择 `report.tex`，编译器选择 **XeLaTeX**。正文使用 `ctexart` 和 TeX Live 自带的 Fandol 中文字体，不依赖服务器字体路径。保留全部子目录即可找到图片和附带文件。

## 本地 TeX Live / MiKTeX

在本目录执行两次：

```bash
xelatex -interaction=nonstopmode -halt-on-error report.tex
xelatex -interaction=nonstopmode -halt-on-error report.tex
```

也可使用 `latexmk -xelatex report.tex`。需安装中文 LaTeX 支持、Fandol 字体及常用数学、表格、插图和超链接宏包。

项目提供 Tectonic 0.17.0 编译入口。从 Sesame 项目根目录可执行：

```bash
__COMPILE_COMMAND__
```

## 项目内更新

`__EXPORT_COMMAND__` 会根据原 Markdown 报告重新生成本目录的 `report.tex`，因此会覆盖对生成稿的直接编辑。仅重新编译已有 `.tex` 时无需运行该脚本。修改 LaTeX 后应重新编译 PDF。

`content_manifest.json` 保存转换时的原表格、图片与哈希；CSV 为原报告的实测数据。正文中的 `runs/`、`datasets/`、`scripts/` 等实验路径均相对 Sesame 项目根目录解释。`source_report.md` 是转换依据，不是编译入口。
'''
    if PAPER_STYLE is not None:
        instructions = instructions.replace(
            '插图使用原报告的高分辨率 PNG，避免部分矢量 PDF 在嵌入时发生中文字体映射错误；曲线与实测数据未改动。',
            '插图按 `paper_style/report_overrides.json` 替换为论文样式的高分辨率 PNG，保留 LaTeX 包内的原图路径。'
            '图注与派生的 `source_report.md` 保持一致；原始 `report.md` 不变。\n\n'
            '## 绘图样式与平滑说明\n\n'+PAPER_STYLE['note']+'\n\n'
            '导出器只复制图像与数据，不计算或修改实测值，也不执行平滑。'
            '`content_manifest.json` 记录覆盖配置、实际文件来源及 SHA256；附带文件保留原相对目录。')
        instructions = instructions.replace(
            '`source_report.md` 是转换依据，不是编译入口。',
            '`source_report.md` 是图注已替换的派生稿，不是编译入口。'
            '省略 `--paper-style` 会恢复默认的原报告图像与图注。')
    substitutions = {'__TABLE_COUNT__':str(len(tables)), '__FIGURE_COUNT__':str(len(figures)),
                     '__REPORT_DATE__':str(report_date), '__EXPORT_DATE__':str(export_date),
                     '__COMPILE_COMMAND__':compile_command,
                     '__EXPORT_COMMAND__':shlex.join(export_command)}
    for placeholder, replacement in substitutions.items():
        instructions = instructions.replace(placeholder, replacement)
    (DEST/'BUILD.md').write_text(instructions)
    print(json.dumps(dict(output=str(DEST),tables=len(tables),figures=len(figures),files=len(COPIED)),ensure_ascii=False))


if __name__ == '__main__': main()
