"""R-gold regression tests for grobX / grobY (xDetails / yDetails).

Every expected number was produced by R 4.4.3 grid on a 5x5-inch
pdf(NULL) device:

    convertX(grobX(<grob>, theta), "inches", valueOnly = TRUE)
    convertY(grobY(<grob>, theta), "inches", valueOnly = TRUE)

and verified to match this port to < 1e-9 inches.  The edge geometry is
class-specific, exactly like R:

    rect / raster        rectEdge of the justified rect (union bbox if >1)
    circle               point ON the circle (union-bbox rectEdge if >1)
    lines/polyline/polygon/points/pathgrob/null
                         hullEdge (chull + polygonEdge) of the locations
    segments             hullEdge of the recycled endpoints
    xspline              hullEdge of the EVALUATED curve (per id group)
    beziergrob           xspline edge of the splinegrob approximation
    curve / functiongrob delegation to the expanded content
    text                 per-label justified+rotated box (polygonEdge)
    everything else      unit(0.5, "npc")   (R xDetails.default)

Excluded by design: the fun_parab case at theta=45.  A hull vertex of
the parabola lies exactly ON the 45-degree ray from the bbox centre, so
polygonEdge's vertex-angle comparison sits on an exact-equality boundary.
R's coordinates (seq + convertX) land ON the boundary: its scan picks the
hull chord, which is exactly parallel to the ray (denom == 0), and R
errors with "polygon edge not found (zero-width or zero-height?)".  This
port's coordinates (linspace + unit conversion) differ in the last ULP,
so the scan picks the adjacent edge and returns a finite point.  Same
algorithm both sides; which side of the boundary the rounding lands on
is not reproducible bit-for-bit across languages.

Text gold values are compared with a loose tolerance (0.1 in): cairo
font metrics differ slightly from R's AFM Helvetica (see _text_bbox).
"""

from __future__ import annotations

import numpy as np
import pytest

import grid_py as g
from grid_py._grob import GList, GTree
from grid_py._size import (
    _chull,
    _circle_edge,
    _polygon_edge,
    _rect_edge,
    grob_x,
    grob_y,
    x_details,
    y_details,
)
from grid_py._draw import grid_newpage
from grid_py._state import get_state
from grid_py._units import Unit, convert_x, convert_y


def _make_cases():
    return {
        "rect1": g.rect_grob(x=0.5, y=0.5, width=0.4, height=0.2),
        "rect_just": g.rect_grob(x=0.3, y=0.4, width=0.35, height=0.25,
                                 just=("left", "bottom")),
        "rect_multi": g.rect_grob(x=[0.3, 0.7], y=[0.3, 0.6],
                                  width=[0.2, 0.3], height=[0.15, 0.4]),
        "circle1": g.circle_grob(x=0.5, y=0.5, r=0.3),
        "circle_mult": g.circle_grob(x=[0.3, 0.7], y=[0.4, 0.6],
                                     r=[0.1, 0.2]),
        "lines3": g.lines_grob(x=[0.1, 0.5, 0.9], y=[0.1, 0.9, 0.2]),
        "polyline5": g.polyline_grob(x=[0.1, 0.3, 0.5, 0.7, 0.9],
                                     y=[0.2, 0.7, 0.3, 0.8, 0.4]),
        "polygon5": g.polygon_grob(x=[0.2, 0.5, 0.8, 0.7, 0.3],
                                   y=[0.3, 0.8, 0.55, 0.2, 0.15]),
        "points4": g.points_grob(x=[0.2, 0.4, 0.6, 0.8],
                                 y=[0.7, 0.2, 0.8, 0.3], pch=1,
                                 default_units="npc"),
        "segments2": g.segments_grob(x0=[0.1, 0.2], y0=[0.1, 0.8],
                                     x1=[0.9, 0.6], y1=[0.5, 0.3]),
        "pathg": g.path_grob(x=[0.1, 0.9, 0.9, 0.1, 0.3, 0.7, 0.7, 0.3],
                             y=[0.1, 0.1, 0.9, 0.9, 0.3, 0.3, 0.7, 0.7],
                             id=[1, 1, 1, 1, 2, 2, 2, 2]),
        "xspl_open": g.xspline_grob(x=[0.1, 0.35, 0.6, 0.9],
                                    y=[0.5, 0.9, 0.1, 0.6], shape=0.7),
        "xspl_closed": g.xspline_grob(x=[0.2, 0.5, 0.8, 0.7, 0.3],
                                      y=[0.3, 0.8, 0.55, 0.2, 0.15],
                                      shape=-0.5, open_=False),
        "xspl_id": g.xspline_grob(
            x=[0.1, 0.3, 0.4, 0.2, 0.6, 0.8, 0.9, 0.7],
            y=[0.1, 0.2, 0.5, 0.7, 0.7, 0.5, 0.2, 0.1],
            shape=0.4, id=[1, 1, 1, 1, 2, 2, 2, 2]),
        "bezier1": g.bezier_grob(x=[0.1, 0.3, 0.7, 0.9],
                                 y=[0.2, 0.8, 0.8, 0.2]),
        "curve1": g.curve_grob(0.2, 0.3, 0.7, 0.8, curvature=0.6, ncp=8,
                               square=False),
        "roundrect1": g.roundrect_grob(x=0.5, y=0.45, width=0.5,
                                       height=0.3, r=Unit(0.2, "inches")),
        "null1": g.null_grob(x=0.3, y=0.7),
        "fun1": g.function_grob(
            fn=lambda t: {"x": t, "y": 0.3 + 0.4 * np.sin(np.pi * t)},
            n=21, range=(0.1, 0.9), units="npc"),
        # theta=45 is excluded from GOLD (fp knife edge, see module doc)
        "fun_parab": g.function_grob(
            fn=lambda t: {"x": t, "y": t ** 2},
            n=21, range=(0.1, 0.9), units="npc"),
        "gtree1": GTree(children=GList(g.rect_grob(width=0.2,
                                                   height=0.2))),
    }


@pytest.fixture
def cases():
    # function-scoped: conftest's autouse _reset_grid_state wipes the
    # renderer before every test, so the page must be recreated per test
    grid_newpage(width=5.0, height=5.0, dpi=100)
    yield _make_cases()
    get_state().reset()


GOLD = [
    ("rect1", 0.0, 3.5, 2.5),
    ("rect1", 17.3, 3.5, 2.81146531595416),
    ("rect1", 45.0, 3.0, 3.0),
    ("rect1", 90.0, 2.5, 3.0),
    ("rect1", 135.0, 2.0, 3.0),
    ("rect1", 213.7, 1.75028162762935, 2.0),
    ("rect1", 270.0, 2.5, 2.0),
    ("rect1", 331.0, 3.40202387763571, 2.0),
    ("rect_just", 0.0, 3.25, 2.625),
    ("rect_just", 17.3, 3.25, 2.89753215145989),
    ("rect_just", 45.0, 3.0, 3.25),
    ("rect_just", 90.0, 2.375, 3.25),
    ("rect_just", 135.0, 1.75, 3.25),
    ("rect_just", 213.7, 1.5, 2.04144754067237),
    ("rect_just", 270.0, 2.375, 2.0),
    ("rect_just", 331.0, 3.25, 2.13997957997883),
    ("rect_multi", 0.0, 4.25, 2.5625),
    ("rect_multi", 17.3, 4.25, 3.06863113842551),
    ("rect_multi", 45.0, 4.0625, 4.0),
    ("rect_multi", 90.0, 2.625, 4.0),
    ("rect_multi", 135.0, 1.1875, 4.0),
    ("rect_multi", 213.7, 1.0, 1.47875971839155),
    ("rect_multi", 270.0, 2.625, 1.125),
    ("rect_multi", 331.0, 4.25, 1.66174779138925),
    ("circle1", 0.0, 4.0, 2.5),
    ("circle1", 17.3, 3.9321411992542, 2.94606231111668),
    ("circle1", 45.0, 3.56066017177982, 3.56066017177982),
    ("circle1", 90.0, 2.5, 4.0),
    ("circle1", 135.0, 1.43933982822018, 3.56066017177982),
    ("circle1", 213.7, 1.25206881680428, 1.667733358828),
    ("circle1", 270.0, 2.5, 1.0),
    ("circle1", 331.0, 3.81192956070909, 1.77278556963049),
    ("circle_mult", 0.0, 4.5, 2.75),
    ("circle_mult", 17.3, 4.5, 3.29506430291978),
    ("circle_mult", 45.0, 4.0, 4.0),
    ("circle_mult", 90.0, 2.75, 4.0),
    ("circle_mult", 135.0, 1.5, 4.0),
    ("circle_mult", 213.7, 1.0, 1.58289508134475),
    ("circle_mult", 270.0, 2.75, 1.5),
    ("circle_mult", 331.0, 4.5, 1.77995915995765),
    ("lines3", 0.0, 3.64285714285714, 2.5),
    ("lines3", 17.3, 3.4701836768834, 2.80217856545406),
    ("lines3", 45.0, 3.22727272727273, 3.22727272727273),
    ("lines3", 90.0, 2.5, 4.5),
    ("lines3", 135.0, 1.83333333333333, 3.16666666666667),
    ("lines3", 213.7, 0.999718213653012, 1.49943642730602),
    ("lines3", 270.0, 2.5, 0.75),
    ("lines3", 331.0, 4.17267302845272, 1.57282220020774),
    ("polyline5", 0.0, 4.25, 2.5),
    ("polyline5", 17.3, 4.01419100941829, 2.97161798116342),
    ("polyline5", 45.0, 3.66666666666667, 3.66666666666667),
    ("polyline5", 90.0, 2.5, 3.75),
    ("polyline5", 135.0, 1.5, 3.5),
    ("polyline5", 213.7, 0.590648277239665, 1.22662069309916),
    ("polyline5", 270.0, 2.5, 1.5),
    ("polyline5", 331.0, 3.74330317829169, 1.81082579457292),
    ("polygon5", 0.0, 3.89285714285714, 2.375),
    ("polygon5", 17.3, 3.91946358952413, 2.81711367539656),
    ("polygon5", 45.0, 3.38636363636364, 3.26136363636364),
    ("polygon5", 90.0, 2.5, 4.0),
    ("polygon5", 135.0, 1.890625, 2.984375),
    ("polygon5", 213.7, 1.05785899459256, 1.41321150811117),
    ("polygon5", 270.0, 2.5, 0.875),
    ("polygon5", 331.0, 3.70242436827877, 1.7084852889757),
    ("points4", 0.0, 3.6, 2.5),
    ("points4", 17.3, 3.47813762253962, 2.80465594365095),
    ("points4", 45.0, 3.28571428571429, 3.28571428571429),
    ("points4", 90.0, 2.5, 3.875),
    ("points4", 135.0, 1.4, 3.6),
    ("points4", 213.7, 1.63164761933671, 1.92088095165822),
    ("points4", 270.0, 2.5, 1.125),
    ("points4", 331.0, 3.9133796541806, 1.71655086454851),
    ("segments2", 0.0, 4.125, 2.25),
    ("segments2", 17.3, 3.99606470939846, 2.71597226740066),
    ("segments2", 45.0, 3.275, 3.025),
    ("segments2", 90.0, 2.5, 3.35714285714286),
    ("segments2", 135.0, 0.96875, 3.78125),
    ("segments2", 213.7, 0.565712989010912, 0.959990923076383),
    ("segments2", 270.0, 2.5, 1.3),
    ("segments2", 331.0, 3.38726853225378, 1.75817902150252),
    ("pathg", 0.0, 4.5, 2.5),
    ("pathg", 17.3, 4.5, 3.12293063190832),
    ("pathg", 45.0, 4.5, 4.5),
    ("pathg", 90.0, 2.5, 4.5),
    ("pathg", 135.0, 0.5, 4.5),
    ("pathg", 213.7, 0.5, 1.16616580725114),
    ("pathg", 270.0, 2.5, 0.5),
    ("pathg", 331.0, 4.5, 1.39138189709446),
    ("xspl_open", 0.0, 4.23236188810393, 2.55425993880194),
    ("xspl_open", 17.3, 4.18383184136651, 3.07871515528684),
    ("xspl_open", 45.0, 3.25556275596908, 3.30982269477102),
    ("xspl_open", 90.0, 2.5, 3.49793221194097),
    ("xspl_open", 135.0, 1.38616216795705, 3.66809777084489),
    ("xspl_open", 213.7, 1.67167300324886, 2.00183450328012),
    ("xspl_open", 270.0, 2.5, 1.64965097712664),
    ("xspl_open", 331.0, 3.7688646223825, 1.85091679354712),
    ("xspl_closed", 0.0, 4.02738688211792, 2.37816898396146),
    ("xspl_closed", 17.3, 3.96465014262466, 2.83433082205756),
    ("xspl_closed", 45.0, 3.51906547232747, 3.39715136087677),
    ("xspl_closed", 90.0, 2.50008309541216, 4.00002604288484),
    ("xspl_closed", 135.0, 1.79037892894432, 3.0878731504293),
    ("xspl_closed", 213.7, 0.969864490672516, 1.35764003527037),
    ("xspl_closed", 270.0, 2.50008309541216, 0.757922719395259),
    ("xspl_closed", 331.0, 3.85327552604654, 1.62808217130345),
    ("xspl_id", 0.0, 4.3897459675624, 2.0),
    ("xspl_id", 17.3, 4.3897459675624, 2.60576047838412),
    ("xspl_id", 45.0, 3.9448729837812, 3.5),
    ("xspl_id", 90.0, 2.4448729837812, 3.5),
    ("xspl_id", 135.0, 0.944872983781203, 3.5),
    ("xspl_id", 213.7, 0.5, 0.702930956839567),
    ("xspl_id", 270.0, 2.4448729837812, 0.5),
    ("xspl_id", 331.0, 4.3897459675624, 0.921939301164123),
    ("bezier1", 0.0, 4.0148145194561, 2.08399445583186),
    ("bezier1", 17.3, 3.74369583633704, 2.47136257244746),
    ("bezier1", 45.0, 3.30647932627049, 2.89047378210235),
    ("bezier1", 90.0, 2.5, 3.16744082957848),
    ("bezier1", 135.0, 1.69415728709965, 2.88983716873222),
    ("bezier1", 213.7, 0.874618881829844, 1.0),
    ("bezier1", 270.0, 2.5, 1.0),
    ("bezier1", 331.0, 4.45557776477014, 1.0),
    ("curve1", 0.0, 3.44511286766328, 2.713023213942),
    ("curve1", 17.3, 3.54988297457321, 3.10637547493856),
    ("curve1", 45.0, 3.51112985076375, 3.93717880210757),
    ("curve1", 90.0, 2.28697426259818, 2.78697426259818),
    ("curve1", 135.0, 2.24999873827009, 2.74999873827009),
    ("curve1", 213.7, 2.06495431545818, 2.56495431545818),
    ("curve1", 270.0, 2.28697426259818, 1.55489695850497),
    ("curve1", 331.0, 3.17517400983434, 2.22068605455093),
    ("roundrect1", 0.0, 3.75, 2.25),
    ("roundrect1", 17.3, 3.75, 2.6393316449427),
    ("roundrect1", 45.0, 3.25, 3.0),
    ("roundrect1", 90.0, 2.5, 3.0),
    ("roundrect1", 135.0, 1.75, 3.0),
    ("roundrect1", 213.7, 1.39011648774502, 1.50979971069305),
    ("roundrect1", 270.0, 2.5, 1.5),
    ("roundrect1", 331.0, 3.7095892654003, 1.5795137216485),
    ("null1", 0.0, 1.5, 3.5),
    ("null1", 17.3, 1.5, 3.5),
    ("null1", 45.0, 1.5, 3.5),
    ("null1", 90.0, 1.5, 3.5),
    ("null1", 135.0, 1.5, 3.5),
    ("null1", 213.7, 1.5, 3.5),
    ("null1", 270.0, 1.5, 3.5),
    ("null1", 331.0, 1.5, 3.5),
    ("fun1", 0.0, 3.86267502308576, 2.80901699437495),
    ("fun1", 17.3, 3.49751278376267, 3.1197076287379),
    ("fun1", 45.0, 3.06436480989636, 3.37338180427131),
    ("fun1", 90.0, 2.5, 3.5),
    ("fun1", 135.0, 1.93563519010364, 3.37338180427131),
    ("fun1", 213.7, 1.46391469137401, 2.11803398874989),
    ("fun1", 270.0, 2.5, 2.11803398874989),
    ("fun1", 331.0, 3.74656634022858, 2.11803398874989),
    ("fun_parab", 0.0, 3.2, 2.05),
    ("fun_parab", 17.3, 3.41556341874718, 2.33516624949616),
    # (fun_parab, 45.0) excluded: R errors here — see module docstring
    ("fun_parab", 90.0, 2.5, 2.05),
    ("fun_parab", 135.0, 2.5, 2.05),
    ("fun_parab", 213.7, 2.5, 2.05),
    ("fun_parab", 270.0, 2.5, 1.25),
    ("fun_parab", 331.0, 2.98338119175624, 1.78205743010749),
    ("gtree1", 0.0, 2.5, 2.5),
    ("gtree1", 17.3, 2.5, 2.5),
    ("gtree1", 45.0, 2.5, 2.5),
    ("gtree1", 90.0, 2.5, 2.5),
    ("gtree1", 135.0, 2.5, 2.5),
    ("gtree1", 213.7, 2.5, 2.5),
    ("gtree1", 270.0, 2.5, 2.5),
    ("gtree1", 331.0, 2.5, 2.5),
]


TEXT_GOLD = [
    ("text1", 0.0, 2.35316666666667, 3.0),
    ("text1", 17.3, 2.19210271663808, 3.05983333333333),
    ("text1", 45.0, 2.05983333333333, 3.05983333333333),
    ("text1", 90.0, 2.0, 3.05983333333333),
    ("text1", 135.0, 1.94016666666667, 3.05983333333333),
    ("text1", 213.7, 1.91028370143964, 2.94016666666667),
    ("text1", 270.0, 2.0, 2.94016666666667),
    ("text1", 331.0, 2.10794219069041, 2.94016666666667),
    ("text_rot", 0.0, 2.60431623327133, 2.5),
    ("text_rot", 17.3, 2.68789575382478, 2.55852301033148),
    ("text_rot", 45.0, 2.70954072340017, 2.70954072340017),
    ("text_rot", 90.0, 2.5, 2.57304301289423),
    ("text_rot", 135.0, 2.45703876659014, 2.54296123340986),
    ("text_rot", 213.7, 2.25714554688608, 2.33803621328767),
    ("text_rot", 270.0, 2.5, 2.42695698710577),
    ("text_rot", 331.0, 2.55822403106991, 2.46772589256588),
    ("text_just", 0.0, 2.108, 1.55983333333333),
    ("text_just", 17.3, 1.99610271663808, 1.61966666666667),
    ("text_just", 45.0, 1.86383333333333, 1.61966666666667),
    ("text_just", 90.0, 1.804, 1.61966666666667),
    ("text_just", 135.0, 1.74416666666667, 1.61966666666667),
    ("text_just", 213.7, 1.71428370143965, 1.5),
    ("text_just", 270.0, 1.804, 1.5),
    ("text_just", 331.0, 1.91194219069041, 1.5),
    ("text_two", 0.0, 3.62883333333333, 2.5),
    ("text_two", 17.3, 3.62883333333333, 2.85317571284902),
    ("text_two", 45.0, 3.55475, 3.55983333333333),
    ("text_two", 90.0, 2.49491666666667, 3.55983333333333),
    ("text_two", 135.0, 1.43508333333333, 3.55983333333333),
    ("text_two", 213.7, 1.361, 1.74377158913609),
    ("text_two", 270.0, 2.49491666666667, 1.44016666666667),
    ("text_two", 331.0, 3.62883333333333, 1.87145972807351),
]


@pytest.mark.parametrize("case,theta,gx,gy",
                         GOLD, ids=[f"{c}-{t}" for c, t, _, _ in GOLD])
def test_grobxy_matches_r(cases, case, theta, gx, gy):
    grob = cases[case]
    px = float(np.atleast_1d(
        convert_x(grob_x(grob, theta), "inches", valueOnly=True))[0])
    py = float(np.atleast_1d(
        convert_y(grob_y(grob, theta), "inches", valueOnly=True))[0])
    assert px == pytest.approx(gx, abs=1e-9)
    assert py == pytest.approx(gy, abs=1e-9)


@pytest.mark.parametrize("case,theta,gx,gy", TEXT_GOLD,
                         ids=[f"{c}-{t}" for c, t, _, _ in TEXT_GOLD])
def test_text_grobxy_near_r(cases, case, theta, gx, gy):
    # cairo vs AFM-Helvetica string metrics: loose tolerance by design
    tcases = {
        "text1": g.text_grob("Hello grid", x=0.4, y=0.6),
        "text_rot": g.text_grob("Rotated", x=0.5, y=0.5, rot=35),
        "text_just": g.text_grob("Justified", x=0.3, y=0.3,
                                 just=("left", "bottom")),
        "text_two": g.text_grob(["one", "two"], x=[0.3, 0.7],
                                y=[0.3, 0.7]),
    }
    grob = tcases[case]
    px = float(np.atleast_1d(
        convert_x(grob_x(grob, theta), "inches", valueOnly=True))[0])
    py = float(np.atleast_1d(
        convert_y(grob_y(grob, theta), "inches", valueOnly=True))[0])
    assert px == pytest.approx(gx, abs=0.1)
    assert py == pytest.approx(gy, abs=0.1)


class TestTextEdgeStructure:
    """Metric-independent invariants of the text edge geometry."""

    def test_left_bottom_just_theta0(self):
        grid_newpage(width=5.0, height=5.0, dpi=100)
        from grid_py._size import _text_label_extent, _resolve_grob_gp
        grob = g.text_grob("Justified", x=0.3, y=0.3,
                           just=("left", "bottom"))
        gp = _resolve_grob_gp(grob)
        w, h = _text_label_extent("Justified", gp, 1.0, 1.2, 12.0)
        # anchor (0.3, 0.3) npc on 5in = (1.5, 1.5)in; box grows right/up
        pts = x_details(grob, 0)
        assert float(pts._values[0]) == pytest.approx(1.5 + w, abs=1e-9)
        pts = y_details(grob, 0)
        assert float(pts._values[0]) == pytest.approx(1.5 + h / 2,
                                                      abs=1e-9)

    def test_rot180_symmetry(self):
        grid_newpage(width=5.0, height=5.0, dpi=100)
        grob = g.text_grob("Sym", x=0.5, y=0.5)
        x0 = float(x_details(grob, 0)._values[0])
        x180 = float(x_details(grob, 180)._values[0])
        # box is centred on the anchor: east/west edges symmetric about
        # x = 2.5in
        assert (x0 - 2.5) == pytest.approx(2.5 - x180, abs=1e-9)


class TestEdgeGeometryUnits:
    """Unit tests of the ported grid.c geometry helpers."""

    def test_rect_edge_special_angles(self):
        assert _rect_edge(0, 0, 4, 2, 0) == (4, 1)
        assert _rect_edge(0, 0, 4, 2, 90) == (2, 2)
        assert _rect_edge(0, 0, 4, 2, 180) == (0, 1)
        assert _rect_edge(0, 0, 4, 2, 270) == (2, 0)

    def test_circle_edge(self):
        ex, ey = _circle_edge(1.0, 2.0, 0.5, 45)
        assert ex == pytest.approx(1.0 + 0.5 * np.cos(np.pi / 4))
        assert ey == pytest.approx(2.0 + 0.5 * np.sin(np.pi / 4))

    def test_polygon_edge_not_found_raises(self):
        # NaN vertices defeat the angle scan -> R: error("polygon edge
        # not found")
        with pytest.raises(ValueError, match="polygon edge not found"):
            _polygon_edge([float("nan")] * 3, [float("nan")] * 3, 33.0)

    def test_chull_matches_r_order(self):
        # R 4.4.3: chull(c(0.2,0.5,0.8,0.7,0.3), c(0.3,0.8,0.55,0.2,0.15))
        # == c(4, 5, 1, 2, 3)  (1-based)
        assert _chull([0.2, 0.5, 0.8, 0.7, 0.3],
                      [0.3, 0.8, 0.55, 0.2, 0.15]) == [3, 4, 0, 1, 2]
        # parabola: 21 hull vertices, R order starts at index 12 (1-based)
        t = np.linspace(0.1, 0.9, 21)
        assert _chull(t, t ** 2) == [11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1,
                                     0, 20, 19, 18, 17, 16, 15, 14, 13,
                                     12]
        # two points: R returns c(min-x, max-x) reversed -> c(2, 1)
        assert _chull([0.3, 0.8], [0.9, 0.1]) == [1, 0]

    def test_gtree_defaults_to_half_npc(self, cases):
        # R has no xDetails.gTree: grobX(gTree, theta) is the centre
        u = x_details(cases["gtree1"], 123.4)
        assert u._units[0] == "npc"
        assert float(u._values[0]) == pytest.approx(0.5)

    def test_vp_carrying_grob_adjusts_to_caller(self):
        # R unit.c:508-527: the edge is transformed to device coords with
        # the grob-context transform ("to allow for viewports in grob"),
        # then adjusted relative to the caller's viewport.
        # R 4.4.3 gold: grobX/grobY at theta=30 on a 5x5in device.
        grid_newpage(width=5.0, height=5.0, dpi=100)
        r1 = g.rect_grob(x=0.5, y=0.5, width=0.4, height=0.2,
                         vp=g.Viewport(x=0.25, y=0.25, width=0.5,
                                       height=0.4,
                                       just=("left", "bottom")))
        px = float(np.atleast_1d(
            convert_x(grob_x(r1, 30), "inches", valueOnly=True))[0])
        py = float(np.atleast_1d(
            convert_y(grob_y(r1, 30), "inches", valueOnly=True))[0])
        assert px == pytest.approx(2.8464101615, abs=1e-9)
        assert py == pytest.approx(2.45, abs=1e-9)

    def test_nested_viewport_caller_relative(self):
        # Evaluated inside a pushed viewport, results are relative to it
        # (R 4.4.3 gold values)
        grid_newpage(width=5.0, height=5.0, dpi=100)
        g.push_viewport(g.Viewport(x=0.1, y=0.1, width=0.6, height=0.8,
                                   just=("left", "bottom")))
        c1 = g.circle_grob(x=0.5, y=0.5, r=0.3)
        px = float(np.atleast_1d(
            convert_x(grob_x(c1, 45), "inches", valueOnly=True))[0])
        assert px == pytest.approx(2.1363961031, abs=1e-9)
        b1 = g.bezier_grob(x=[0.1, 0.3, 0.7, 0.9], y=[0.2, 0.8, 0.8, 0.2])
        py = float(np.atleast_1d(
            convert_y(grob_y(b1, 90), "inches", valueOnly=True))[0])
        assert py == pytest.approx(2.5344158929, abs=1e-9)
        g.pop_viewport()

    def test_no_renderer_returns_default(self):
        get_state().reset()
        u = x_details(g.rect_grob(width=0.4, height=0.2), 0)
        assert u._units[0] == "npc"
        assert float(u._values[0]) == pytest.approx(0.5)
