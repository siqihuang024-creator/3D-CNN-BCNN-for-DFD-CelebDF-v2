# -*- coding: utf-8 -*-
"""Build 实验方案_V2.html from 实验方案_V2.md plus the SVG figures in figs.py.

    python _build/build.py

A line `[[FIG:name]]` on its own paragraph in the markdown is replaced by
figure `name` from figs.FIGURES; figures are numbered in order of appearance.
The text 【图:name】 anywhere else becomes a reference "图 N".
"""
import os
import re
import sys

import markdown

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
from figs import FIGURES  # noqa: E402

SRC = os.path.join(ROOT, "实验方案_V2.md")
DST = os.path.join(ROOT, "实验方案_V2.html")
TITLE = "3D-CNN + BCNN V2 实验方案"

CSS = r"""
:root{
  --bg:#ffffff; --fg:#1a1a1a; --muted:#5a6270; --rule:#e2e5ea;
  --accent:#1f5fa9; --accent-soft:#eef4fb; --code-bg:#f5f6f8; --th-bg:#f0f3f7;
  --line:#3b4250; --box-bg:#ffffff; --hl:#c2410c; --hl-bg:#fff3ea; --frz-bg:#eef1f5;
}
@media (prefers-color-scheme: dark){
  :root:not([data-theme="light"]){
    --bg:#15181d; --fg:#e6e8eb; --muted:#9aa3b0; --rule:#2c313a;
    --accent:#6fb0f0; --accent-soft:#1c2733; --code-bg:#1d2128; --th-bg:#222731;
    --line:#aab3c0; --box-bg:#1b1f25; --hl:#fb923c; --hl-bg:#3a2416; --frz-bg:#232830;
  }
}
:root[data-theme="dark"]{
  --bg:#15181d; --fg:#e6e8eb; --muted:#9aa3b0; --rule:#2c313a;
  --accent:#6fb0f0; --accent-soft:#1c2733; --code-bg:#1d2128; --th-bg:#222731;
  --line:#aab3c0; --box-bg:#1b1f25; --hl:#fb923c; --hl-bg:#3a2416; --frz-bg:#232830;
}
*{box-sizing:border-box}
html{-webkit-text-size-adjust:100%}
body{margin:0;background:var(--bg);color:var(--fg);
  font-family:-apple-system,BlinkMacSystemFont,"Segoe UI","Microsoft YaHei","PingFang SC",
              "Hiragino Sans GB",sans-serif;font-size:16px;line-height:1.75}
.wrap{max-width:980px;margin:0 auto;padding:48px 16px 96px}
h1{font-size:1.9rem;line-height:1.3;margin:0 0 .3em;letter-spacing:-.01em}
h2{font-size:1.4rem;margin:2.6em 0 .6em;padding-top:.7em;border-top:1px solid var(--rule)}
h3{font-size:1.12rem;margin:1.9em 0 .5em;color:var(--accent)}
h4{font-size:1rem;margin:1.5em 0 .4em}
h1+p{color:var(--muted);font-size:.92rem;margin-top:0}
p,li{overflow-wrap:break-word}
a{color:var(--accent)}
hr{border:0;border-top:1px solid var(--rule);margin:2.5em 0}
code{background:var(--code-bg);padding:.13em .38em;border-radius:4px;
  font-family:ui-monospace,SFMono-Regular,Consolas,"Liberation Mono",monospace;font-size:.86em}
pre{background:var(--code-bg);border:1px solid var(--rule);border-radius:8px;
  padding:14px 16px;overflow-x:auto;line-height:1.55}
pre code{background:none;padding:0;font-size:.84rem}
blockquote{margin:1.4em 0;padding:.7em 1.1em;background:var(--accent-soft);
  border-left:3px solid var(--accent);border-radius:0 6px 6px 0}
blockquote p{margin:.4em 0}
.tablewrap{overflow-x:auto;margin:1.3em 0}
table{border-collapse:collapse;width:100%;font-size:.9rem}
th,td{border:1px solid var(--rule);padding:7px 11px;text-align:left;vertical-align:top}
th{background:var(--th-bg);font-weight:600;white-space:nowrap}
tbody tr:nth-child(even){background:color-mix(in srgb,var(--th-bg) 45%,transparent)}
strong{font-weight:650}
ul,ol{padding-left:1.5em}
li{margin:.3em 0}
.toc{background:var(--code-bg);border:1px solid var(--rule);border-radius:8px;
  padding:12px 20px;margin:1.5em 0;font-size:.92rem}
.toc ul{margin:.2em 0;padding-left:1.2em}
.toc>ul{padding-left:.2em;list-style:none}
/* figures */
figure.fig{margin:1.8em 0 2em}
.figscroll{overflow-x:auto}
figure.fig svg{min-width:760px}
figcaption{font-size:.88rem;color:var(--muted);margin-top:.6em;line-height:1.65}
figcaption b{color:var(--fg)}
.dg{display:block;width:100%;height:auto;font-family:inherit}
.dg text{fill:var(--fg);font-size:12px}
.dg .t{font-weight:600}
.dg .tn{font-weight:400}
.dg .th{font-weight:700;font-size:12.5px}
.dg .ts{font-size:11px;fill:var(--muted)}
.dg .tl{font-size:11px;fill:var(--muted)}
.dg .tm{font-family:ui-monospace,SFMono-Regular,Consolas,monospace;font-size:11px}
.dg .hlt{fill:var(--hl)}
.dg .box{fill:var(--box-bg);stroke:var(--line);stroke-width:1.2}
.dg .box.hl{fill:var(--hl-bg);stroke:var(--hl);stroke-width:1.7}
.dg .box.frz{fill:var(--frz-bg);stroke-dasharray:5 3}
.dg .box.gh{fill:none;stroke:var(--muted);stroke-dasharray:4 3}
.dg .ln{fill:none;stroke:var(--line);stroke-width:1.3}
.dg .ln.hl{stroke:var(--hl);stroke-width:1.6}
.dg .ln.thin{stroke-width:1.1;opacity:.85}
.dg .ln.dash{stroke-dasharray:5 4}
.dg .ln.faint{stroke:var(--muted);stroke-width:1;opacity:.6}
.dg .mk{fill:var(--line)}
.dg .mk-hl{fill:var(--hl)}
.dg .cell{fill:var(--hl-bg);stroke:none}
.dg .face{fill:none;stroke:var(--hl);stroke-width:1.8}
.dg .dot{fill:var(--line)}
.dg .dot.hl{fill:var(--hl)}
.dg .dot.o{fill:var(--bg);stroke:var(--muted);stroke-width:1}
@media (max-width:640px){
  .wrap{padding:28px 16px 64px}
  body{font-size:15px}
  h1{font-size:1.5rem}
  table{font-size:.82rem}
  th,td{padding:6px 8px;white-space:normal}
}
@media print{
  body{background:#fff;color:#000}
  .wrap{max-width:none;padding:0}
  h2{break-after:avoid}
  table,pre,blockquote,figure{break-inside:avoid}
  figure.fig svg{min-width:0}
}
"""


def main():
    md_text = open(SRC, encoding="utf-8").read()

    # number figures in order of appearance
    order = re.findall(r"^\[\[FIG:(\w+)\]\]\s*$", md_text, flags=re.M)
    num = {name: i + 1 for i, name in enumerate(order)}
    unknown = [n for n in order if n not in FIGURES]
    if unknown:
        raise SystemExit("unknown figures: {}".format(unknown))

    # in-text references
    def ref(m):
        n = m.group(1)
        if n not in num:
            raise SystemExit("reference to undeclared figure: " + n)
        return "图 {}".format(num[n])
    md_text = re.sub(r"【图:(\w+)】", ref, md_text)

    body = markdown.markdown(md_text, extensions=["tables", "fenced_code", "toc", "attr_list"],
                             extension_configs={"toc": {"toc_depth": "2-2"}})

    warnings = []

    def fig_html(m):
        name = m.group(1)
        f, cap = FIGURES[name]()
        warnings.extend(f.warn)
        return ('<figure class="fig" id="fig-{n}"><div class="figscroll">{svg}</div>'
                '<figcaption><b>图 {i} ·</b> {cap}</figcaption></figure>').format(
            n=name, svg=f.render(), i=num[name], cap=cap)

    body = re.sub(r"<p>\[\[FIG:(\w+)\]\]</p>", fig_html, body)

    page = """<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title}</title>
<style>{css}</style>
</head>
<body>
<div class="wrap">
{body}
</div>
<script>
document.querySelectorAll('table').forEach(function (t) {{
  var w = document.createElement('div'); w.className = 'tablewrap';
  t.parentNode.insertBefore(w, t); w.appendChild(t);
}});
</script>
</body>
</html>
""".format(title=TITLE, css=CSS, body=body)
    open(DST, "w", encoding="utf-8").write(page)
    print("figures:", ", ".join("{}={}".format(k, v) for k, v in num.items()))
    print("tables:", body.count("<table>"), "| bytes:", len(page.encode("utf-8")))
    for w in warnings:
        print("WARN", w)


if __name__ == "__main__":
    main()
