"""Build a single-page A4 project summary PDF.

Usage:
    python scripts/build_one_page_summary.py

Requires reportlab and pypdf. The Chinese text uses the built-in
STSong-Light CID font, so no external font file is needed.
"""

from __future__ import annotations

from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.cidfonts import UnicodeCIDFont
from reportlab.platypus import (
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT = PROJECT_ROOT / "outputs" / "pdf" / "one_page_summary.pdf"

pdfmetrics.registerFont(UnicodeCIDFont("STSong-Light"))

GRAY = colors.HexColor("#4a4a4a")
BLUE = colors.HexColor("#1f4e79")
LIGHT = colors.HexColor("#dce6f1")


def style(font_size=9.6, leading=13.4, text_color=GRAY, **kwargs) -> ParagraphStyle:
    return ParagraphStyle(
        name=f"s{font_size}",
        fontName="STSong-Light",
        fontSize=font_size,
        leading=leading,
        textColor=text_color,
        **kwargs,
    )


def clean(text: str) -> str:
    """Replace glyphs that the STSong CID font may not cover."""
    return (
        text.replace("\u2212", "-")  # minus sign
        .replace("\u2192", "->")  # arrow
        .replace("\u03b1", "a")  # alpha
        .replace("\uff5c", " | ")  # full-width bar
    )


class CleanParagraph(Paragraph):
    def __init__(self, text: str, *args, **kwargs) -> None:
        super().__init__(clean(text), *args, **kwargs)


def build() -> None:
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    doc = SimpleDocTemplate(
        str(OUTPUT),
        pagesize=A4,
        leftMargin=15 * mm,
        rightMargin=15 * mm,
        topMargin=13 * mm,
        bottomMargin=12 * mm,
        title="One-Page Summary - Options Volatility & Risk Platform",
        author="Wan Yiwen",
    )
    title = style(
        17,
        21,
        BLUE,
        spaceAfter=2,
        alignment=0,
    )
    subtitle = style(9.4, 13, GRAY, spaceAfter=6)
    heading = style(12, 15, BLUE, spaceBefore=7, spaceAfter=2)
    body = style(9.6, 13.4, spaceAfter=2)
    bullet = style(9.5, 13.1, leftIndent=8, bulletIndent=0, spaceAfter=1)
    small = style(8.6, 11.4, GRAY, spaceBefore=2)

    story = [
        CleanParagraph("期权定价、波动率曲面与风险验证平台", title),
        CleanParagraph(
            "一个端到端的期权定价、回测、动态对冲与做市仿真研究平台，强调"
            "数值正确性、可复现性与工程质量。（研究型仿真平台，非生产交易"
            "系统；历史回测用真实 SPY 标的价格与合成期权报价）",
            subtitle,
        ),
        CleanParagraph("核心亮点", heading),
        CleanParagraph(
            "1. 六种定价模型全部通过收敛与退化验证：CRR 步数 100→800 误差 "
            "1.74e-2→2.18e-3；Heston σ_v→0 半解析定价精确回到 Black-Scholes"
            "（误差 0）；修复了 Heston CF 缺对数分支项与 Euler 漂移偏差两个"
            "真实数值 bug。",
            bullet,
        ),
        CleanParagraph(
            "2. 真实 SPY 期权快照 → SVI 曲面：1500 条报价清洗后 872 条活跃、"
            "734 条成功反解 IV；分到期校准 RMSE 3.9e-5~1.7e-4。",
            bullet,
        ),
        CleanParagraph(
            "3. 严格样本外回测 + PnL 逐笔对账：信号滞后一日、训练/验证/测试"
            "隔离、测试集锁定（Bonferroni α=0.0125）；17 笔滚动跨式对账最大"
            "差异 3.456e-11。",
            bullet,
        ),
        CleanParagraph(
            "4. 事件驱动做市 + C++ 加速：多合约 Greeks 感知报价与连续时间 "
            "Actor-Critic；C++ 内核一致性 1e-9，独立基准约 40.8x"
            "（批量 vs Python 标量约 3.8x，口径如实区分）。",
            bullet,
        ),
        CleanParagraph("关键结果（2026-09-04 实测）", heading),
    ]

    rows = [
        ["结果", "数值"],
        ["legacy 长跨式（33 笔）", "PnL −8038.86，Sharpe-like −1.74"],
        ["波动率过滤（阈值 1.30）", "6 笔，PnL +1357.48，Sharpe-like +0.32"],
        ["严格样本外（测试段）", "5 笔，PnL +1731.79，年化 Sharpe 1.28"],
        ["VaR 回测（1194 天）", "87 例外；Kupiec p=0.0007"],
        ["测试与覆盖率", "300+ 测试；86%（门槛 80%）"],
    ]
    table = Table(
        [[clean(cell) for cell in row] for row in rows],
        colWidths=[65 * mm, 105 * mm],
    )
    table.setStyle(
        TableStyle(
            [
                ("FONTNAME", (0, 0), (-1, -1), "STSong-Light"),
                ("FONTSIZE", (0, 0), (-1, -1), 8.8),
                ("BACKGROUND", (0, 0), (-1, 0), LIGHT),
                ("TEXTCOLOR", (0, 0), (-1, 0), BLUE),
                ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#b7c6d6")),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("TOPPADDING", (0, 0), (-1, -1), 2),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
            ]
        )
    )
    story += [table, Spacer(1, 2)]
    story += [
        CleanParagraph("负结果的科学处理", heading),
        CleanParagraph(
            "系统性买长跨式亏钱符合期权经济学（实现波动率需超过隐含波动率"
            "+ 成本）；若它稳定盈利反而说明仿真有 bug。波动率状态过滤把 "
            "Sharpe-like 从 −1.74 改善到 +0.32，方向与期权理论一致；样本小"
            "（6–33 笔），报告如实披露置信区间宽、统计功效低。",
            body,
        ),
        CleanParagraph("技术栈", heading),
        CleanParagraph(
            "Python（pandas/NumPy/SciPy）、C++17（ctypes 绑定）、FastAPI、"
            "SQLite/PostgreSQL、Streamlit、Docker、GitHub Actions、Hypothesis、"
            "pytest-cov。",
            body,
        ),
        CleanParagraph("链接", heading),
        CleanParagraph(
            "GitHub: github.com/burger56487/options-volatility-backtester | "
            "研究报告、架构图与复现指南见仓库 report/ 与 docs/，全部数字可复现。",
            body,
        ),
        Spacer(1, 4),
        CleanParagraph(
            "本材料为研究项目一页纸摘要，不构成投资建议。",
            small,
        ),
    ]
    doc.build(story)
    print(f"saved {OUTPUT}")


if __name__ == "__main__":
    build()
