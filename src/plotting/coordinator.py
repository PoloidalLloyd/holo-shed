"""Routes redraw requests to plotting modules."""

from __future__ import annotations

from src.plotting import monitor_2d, poloidal_2d, polygon_2d, profiles_1d, radial_2d, time_history


class PlotCoordinator:
    """Delegates redraw work out of MainWindow."""

    @staticmethod
    def redraw_profiles(win) -> None:
        profiles_1d.redraw_profiles(win)

    @staticmethod
    def redraw_time_history(win) -> None:
        time_history.redraw_time_history_impl(win)

    @staticmethod
    def redraw_2d_current_tab(win) -> None:
        idx = int(win.plot_tabs.currentIndex())
        if idx == 0:
            poloidal_2d.redraw_poloidal(win)
        elif idx == 1:
            radial_2d.redraw_radial(win)
        elif idx == 2:
            polygon_2d.redraw_polygon(win)
        elif idx == 3:
            monitor_2d.redraw_monitor(win)

    @staticmethod
    def redraw_2d_poloidal(win) -> None:
        poloidal_2d.redraw_poloidal(win)

    @staticmethod
    def redraw_2d_radial(win) -> None:
        radial_2d.redraw_radial(win)

    @staticmethod
    def redraw_2d_polygon(win) -> None:
        polygon_2d.redraw_polygon(win)

    @staticmethod
    def redraw_2d_monitor(win) -> None:
        monitor_2d.redraw_monitor(win)
