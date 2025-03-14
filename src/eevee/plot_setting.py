# plot_setting.py

import matplotlib
import matplotlib.font_manager as fm
from cycler import cycler
import numpy as np


helvetica_bold_path = '/Users/aracho/Dropbox/Resources/Fonts-downloaded/Helvetica/Helvetica-Bold.ttf'

# 폰트 등록
try:
    fm.fontManager.addfont(helvetica_bold_path)
    helvetica_bold_prop = fm.FontProperties(fname=helvetica_bold_path)
    print(f"Helvetica Bold font registered successfully from: {helvetica_bold_path}")
    helvetica_available = True
except Exception as e:
    print(f"Error loading Helvetica Bold font: {e}")
    helvetica_available = False

# 이미지와 일치하는 기본 설정
linewidth = 1.0  # 선 굵기 증가
majorticksize = 4.0  # 주요 눈금 크기 증가
minorticksize = 2.0
labelsize = 12  # 레이블 크기 증가
fontsize = 12  # 폰트 크기 증가
figsize = (3.5, 2.8)  # 그림 크기 유지

    # color_dict = {
    #     0: '#00b7eb',    # light blue
    #     25: '#1f77b4',   # blue
    #     50: '#ff7f7f',   # pink
    #     70: '#ff4444',   # red
    #     90: '#ff0000',   # dark red
    #     120: '#b22222',  # dark red
    #     140: '#8b0000'   # dark red
    # }

# 이미지에 맞는 색상 설정 (파란색과 빨간색이 주요 색상)
colors = [
    '#1A6FDF',  # 파란색 (0°C 전해질 온도)
    '#D32F2F',  # 빨간색 (25°C 전해질 온도)
    '#515151',  # 회색
    '#37AD6B',  # 녹색
    '#B177DE',  # 보라색
    '#FEC211',  # 노란색
    '#999999',  # 회색
    '#FF4081',  # 핫핑크
    '#FB6501',  # 주황색
    '#6699CC',  # 연한 파란색
]

preamble = r"""
              \usepackage{color}
              \usepackage[tx]{sfmath}
              \usepackage{helvet}
              \usepackage{sansmath}
           """

options = {
    'font.size': fontsize,
    
    # 축 설정
    'axes.labelsize': labelsize,
    'axes.linewidth': linewidth,  # 축 선 굵기 증가
    'axes.prop_cycle': cycler('color', colors),
    'axes.titlesize': fontsize,
    'axes.edgecolor': 'black',
    'axes.grid': False,  # 그리드 없음
    
    # 저장 설정
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.01,
    
    # 그림 설정
    'figure.figsize': figsize,
    'figure.dpi': 300,
    'figure.edgecolor': 'white',
    'figure.facecolor': 'white',
    'figure.frameon': True,
    
    # 폰트 설정 - Helvetica 우선
    'font.family': ['sans-serif'],
    'font.sans-serif': ['Helvetica', 'Arial', 'DejaVu Sans', 'Bitstream Vera Sans', 'Computer Modern Sans Serif'],
    'font.weight': 'bold',  # 기본 폰트 굵기
    
    # LaTeX 설정
    'text.usetex': False,  # 기본적으로 TeX 비활성화
    'text.latex.preamble': preamble,
    
    # 그리드 설정
    'grid.alpha': 0,  # 그리드 없음
    'grid.color': '93939c',
    'grid.linestyle': '--',
    'grid.linewidth': linewidth * 0.5,
    
    # 범례 설정
    'legend.edgecolor': '0.8',
    'legend.fancybox': False,
    'legend.frameon': False,  # 범례 테두리 없음
    'legend.fontsize': labelsize,
    'legend.title_fontsize': labelsize,
    'legend.markerscale': 1.0,
    
    # 선 및 마커 설정
    'lines.markeredgecolor': 'auto',  # 마커 테두리 색상 자동
    'lines.markersize': 6,  # 마커 크기 증가
    'lines.linewidth': linewidth * 1.5,  # 선 굵기 증가
    'lines.markeredgewidth': 0.8,  # 마커 테두리 굵기
    'lines.solid_capstyle': 'round',  # 선 끝 모양
    
    # 패치 설정
    'patch.linewidth': linewidth,
    'patch.facecolor': 'none',
    'patch.force_edgecolor': False,
    'scatter.edgecolors': 'face',
    
    # X축 눈금 설정
    'xtick.labelsize': labelsize,
    'xtick.color': 'black',
    'xtick.direction': 'in',  # 안쪽 방향 눈금
    'xtick.minor.visible': True,  # 보조 눈금 표시
    'xtick.major.size': majorticksize,
    'xtick.minor.size': minorticksize,
    'xtick.major.top': True,  # 위쪽 눈금 표시
    'xtick.major.bottom': True,  # 아래쪽 눈금 표시
    'xtick.major.width': linewidth,
    'xtick.minor.width': linewidth * 0.8,
    'xtick.minor.bottom': True,
    'xtick.minor.top': True,
    'xtick.alignment': 'center',
    
    # Y축 눈금 설정
    'ytick.labelsize': labelsize,
    'ytick.color': 'black',
    'ytick.direction': 'in',  # 안쪽 방향 눈금
    'ytick.minor.visible': True,  # 보조 눈금 표시
    'ytick.major.right': True,  # 오른쪽 눈금 표시
    'ytick.major.left': True,  # 왼쪽 눈금 표시
    'ytick.major.size': majorticksize,
    'ytick.minor.size': minorticksize,
    'ytick.major.width': linewidth,
    'ytick.minor.width': linewidth * 0.8,
    'ytick.minor.pad': 3.4,
    'ytick.minor.right': True,
    'ytick.minor.left': True,
    
    # 테두리(스파인) 설정
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.spines.left': True,
    'axes.spines.bottom': True,
}

matplotlib.rcParams.update(options)

from matplotlib.ticker import AutoMinorLocator
from matplotlib import rcParams


def pretty_plot(width=None, height=None, plt=None, dpi=None):
    if plt is None:
        plt = matplotlib.pyplot
        if width is None:
            width = matplotlib.rcParams["figure.figsize"][0]
        if height is None:
            height = matplotlib.rcParams["figure.figsize"][1]
        if dpi is not None:
            matplotlib.rcParams["figure.dpi"] = dpi

        fig = plt.figure(figsize=(width, height))
        ax = fig.add_subplot(1, 1, 1)
        matplotlib.rcParams["xtick.minor.visible"] = True
        matplotlib.rcParams["ytick.minor.visible"] = True
        ax.yaxis.set_minor_locator(AutoMinorLocator(2))
        ax.xaxis.set_minor_locator(AutoMinorLocator(2))
        
        # 이미지와 같은 스타일을 위한 추가 설정
        ax.spines['top'].set_visible(True)
        ax.spines['right'].set_visible(True)
        ax.spines['bottom'].set_visible(True)
        ax.spines['left'].set_visible(True)
        
    return plt

def pretty_subplot(
    nrows=None,
    ncols=None,
    width=None,
    height=None,
    sharex=False,
    sharey=True,
    dpi=None,
    plt=None,
    gridspec_kw=None,
    total_num=None,
    ):
    if nrows is None and ncols is None:
        if total_num == 1:
            nrows, ncols = 1, 1
        elif total_num > 1:
            nrows, ncols = 1, total_num
        elif total_num == 4:
            nrows, ncols = 2, 2
        elif total_num > 4:
            nrows, ncols = 2, total_num // 2
        elif total_num == 9:
            nrows, ncols = 3, 3
        elif total_num > 9:
            nrows, ncols = total_num // 3, 3
        else:
            raise ValueError("total_num must be a positive integer")
    elif nrows is None:
        nrows = total_num // ncols
    elif ncols is None:
        ncols = total_num // nrows
    
    if width is None:
        width = rcParams["figure.figsize"][0]*ncols
    if height is None:
        height = rcParams["figure.figsize"][1]*nrows


    plt = matplotlib.pyplot
    fig, axes = plt.subplots(
        nrows,
        ncols,
        sharex=sharex,
        sharey=sharey,
        dpi=dpi,
        figsize=(width, height),
        facecolor="w",
        gridspec_kw=gridspec_kw,
    )
    # if nrows is None and ncols is None:
    #     if total_num == 1:
    #         axes = [axes]
    
    # for ax in axes:
    #     ax.yaxis.set_minor_locator(AutoMinorLocator(2))
    #     ax.xaxis.set_minor_locator(AutoMinorLocator(2))

    axs = axs.flatten() if isinstance(axes, np.ndarray) else [axes]

    return fig, axes

def draw_themed_line(y, ax, orientation="horizontal"):
    """Draw a horizontal line using the theme settings
    Args:
        y (float): Position of line in data coordinates
        ax (Axes): Matplotlib Axes on which line is drawn
    """

    # Note to future developers: feel free to add plenty more optional
    # arguments to this to mess with linestyle, zorder etc.
    # Just .update() the options dict

    themed_line_options = dict(
        color=rcParams["grid.color"],
        linestyle="--",
        dashes=(5, 2),
        zorder=0,
        linewidth=rcParams["ytick.major.width"],
    )

    if orientation == "horizontal":
        ax.axhline(y, **themed_line_options)
    elif orientation == "vertical":
        ax.axvline(y, **themed_line_options)
    else:
        raise ValueError(f'Line orientation "{orientation}" not supported')

# 과학 논문 스타일 설정 함수
def set_scientific_paper_style(ax, title=None, xlabel=None, ylabel=None, legend_title=None):
    """
    이미지와 같은 과학 논문 스타일의 그래프를 설정합니다.
    """
    # Helvetica Bold 폰트가 등록되었다면 사용
    if helvetica_available:
        # 제목 설정
        if title:
            ax.set_title(title, fontproperties=helvetica_bold_prop)
        
        # 축 레이블 설정
        if xlabel:
            ax.set_xlabel(xlabel, fontproperties=helvetica_bold_prop)
        if ylabel:
            ax.set_ylabel(ylabel, fontproperties=helvetica_bold_prop)
            
        # 범례 설정
        if ax.get_legend():
            if legend_title:
                ax.get_legend().set_title(legend_title)
                # 범례 제목에 폰트 적용
                ax.get_legend()._legend_title_box.get_texts()[0].set_fontproperties(helvetica_bold_prop)
            
            # 범례 항목에 폰트 적용
            for text in ax.get_legend().get_texts():
                text.set_fontproperties(helvetica_bold_prop)
    else:
        # 기본 설정 사용
        if title:
            ax.set_title(title)
        if xlabel:
            ax.set_xlabel(xlabel)
        if ylabel:
            ax.set_ylabel(ylabel)
        
        # 범례 설정
        if ax.get_legend():
            if legend_title:
                ax.get_legend().set_title(legend_title)
            ax.get_legend().get_frame().set_linewidth(0.0)
    
    # 그리드 제거
    ax.grid(False)
    
    # 테두리 설정
    for spine in ax.spines.values():
        spine.set_linewidth(linewidth)
    
    return ax