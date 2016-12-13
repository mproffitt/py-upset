"""
Compute the visualisation of intersections between sets

:author: Leo. <leo@opensignal.com>
:author: Martin Proffitt <mproffitt@jitsc.co.uk>

:version: 2.0

"""
__author__ = [
    'leo@opensignal.com',
    'mproffitt@jitsc.co.uk'
]

from functools import partial
from itertools import product
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib.patches import Rectangle

from .resources import DataExtractor
from .resources import GraphStore
from .resources import GridSpecStore
from .resources import Colours

class UpSetPlot():
    """
    Main library class for plotting intersection data
    """
    # pylint: disable=too-many-instance-attributes
    # This is a complex class which art some point needs breaking down

    _graph_store = None
    _grid_spec = None
    _data = None
    _additional_plots = None
    _colours = None
    _query_to_zorder = None
    _x_values = None
    _y_values = None
    _row_map = None

    _figure = None

    WIDTH = 17
    HEIGHT = 10.5
    GRIDSPEC_ROWS = 3
    GRIDSPEC_COLS = 1
    GRIDSPEC_SPACE = .4
    SETSIZE_WIDTH = 3
    DEFAULT_ALPHA = .3
    GRID_VERTICAL_OFFSET = 5
    GRID_HORIZONTAL_OFFSET = 5
    MATRIX_OFFSET = 4
    GAP_DIVISOR = 500.0
    GAP_MULTIPLIER = 20

    def __init__(self, data_extractor, additional_plots=False, highlight=None):
        """
        Generates figures and axes.

        :param data_extractor DataExtractor: Primary plot data
        :param additional_plots: list of dictionaries as specified in plot()
        :param query: list of tuples as specified in plot()
        """

        if not isinstance(data_extractor, DataExtractor):
            raise RuntimeError('\'data_extractor\' must be an instance of pyupset.DataExtractor')

        self._highlight = [] if highlight is None else highlight
        self._additional_plots = additional_plots
        self._data = data_extractor
        self._row_map = {}

        if not self._data.ready:
            self._data.build()

        self._graph_store = GraphStore()
        self._grid_spec = GridSpecStore()
        self._colours = Colours()
        self._setup_colours()
        self._setup_zorder()

        # set figure properties
        self._x_values, self._y_values = self._create_coordinates(
            self._data.rows,
            self._data.columns
        )

        self._prepare_figure()
        if additional_plots:
            self._prepare_additional_plots()

        self._standard_graph_settings = {
            'scatter': {
                'alpha': self.DEFAULT_ALPHA,
                'edgecolor': None
            },
            'hist': {
                'histtype': 'stepfilled',
                'alpha': self.DEFAULT_ALPHA,
                'lw': 0
            }
        }

    def plot(self):
        """
        Creates the main graph comprising bar plot of base set sizes, bar plot of intersection sizes and intersection
        matrix.

        :return: Dictionary containing figure and axes references.
        """
        ylim = self._base_sets_plot()
        self._table_names_plot(ylim)
        xlim = self._inters_sizes_plot()
        self._inters_matrix(xlim, ylim)
        if self._additional_plots:
            self._plot_additional()
        return self._graph_store

    def save(self, filename):
        """
        Saves the main plot to file

        :param string filename
        """
        self._graph_store.figure.savefig(filename=filename)

    @staticmethod
    def _create_coordinates(rows, cols):
        """
        Creates the x, y coordinates shared by the main plots.

        :param rows: number of rows of intersection matrix
        :param cols: number of columns of intersection matrix
        :return: arrays with x and y coordinates
        """
        x_values = (np.arange(cols) + 1)
        y_values = (np.arange(rows) + 1)
        return x_values, y_values

    def _prepare_figure(self):
        """
        Prepares the figure, axes (and their grid) taking into account the additional plots.l

        :return: references to the newly created figure and axes
        """

        # oh joy. Magic numbers...
        self._graph_store.figure = plt.figure(
            figsize=(
                self.WIDTH,
                self.HEIGHT
            )
        )

        self._grid_spec.main_gs = gridspec.GridSpec(
            UpSetPlot.GRIDSPEC_ROWS,
            UpSetPlot.GRIDSPEC_COLS,
            hspace=UpSetPlot.GRIDSPEC_SPACE
        )

        self._grid_spec.top_gs = (
            self._grid_spec.main_gs[:2, 0]
            if self._additional_plots
            else gridspec.GridSpec(1, 1)[0, 0]
        )
        if self._additional_plots:
            self._grid_spec.bottom_gs = self._grid_spec.main_gs[2, 0]

        gs_top = gridspec.GridSpecFromSubplotSpec(
            (self._data.rows * self.GRID_VERTICAL_OFFSET),
            (self._data.columns + self.GRID_HORIZONTAL_OFFSET),
            subplot_spec=self._grid_spec.top_gs,
            wspace=.1,
            hspace=.2
        )

        tablesize_w = self.SETSIZE_WIDTH + 2
        intmatrix_w = tablesize_w + self._data.columns
        intbars_w = tablesize_w + self._data.columns

        self._graph_store.base_set_size = plt.subplot(
            gs_top[
                -1: -self._data.rows,
                0: self.SETSIZE_WIDTH
            ]
        )

        self._graph_store.names = plt.subplot(
            gs_top[
                -1: -self._data.rows,
                self.SETSIZE_WIDTH: tablesize_w
            ]
        )

        self._graph_store.intersection_matrix = plt.subplot(
            gs_top[-1: -self._data.rows, tablesize_w: intmatrix_w]
        )

        self._graph_store.intersection_bars = plt.subplot(
            gs_top[
                :(self._data.rows * self.MATRIX_OFFSET) - 1,
                tablesize_w:intbars_w
            ] # because, magic numbers
        )

    def _prepare_additional_plots(self):
        """ Prepare additional plots if requested """
        additional_axes = []
        num_plots = len(self._additional_plots)

        num_bot_rows = int(np.ceil(num_plots / 2))
        num_bot_cols = 2

        gs_bottom = gridspec.GridSpecFromSubplotSpec(
            num_bot_rows,
            num_bot_cols,
            subplot_spec=self._grid_spec.bottom_gs,
            wspace=.15, # magic number?
            hspace=.2   # magic number...
        )

        for rows, cols in product(range(num_bot_rows), range(num_bot_cols)):
            if (rows + cols + 1) > num_plots:
                break

            additional_axes.append(
                plt.subplot(gs_bottom[rows, cols])
            )
        self.additional_plots_axes = additional_axes

    def _color_for_query(self, query):
        """
        Helper function that returns the standard dark grey for non-queried intersections, and the color assigned to
        a query when the class was instantiated otherwise
        :param query: frozenset.
        :return: color as length 4 array.
        """
        return self._colours.standard.get(
            query,
            self._colours.greys[1]
        )

    def _zorder_for_query(self, query):
        """
        Helper function that returns 0 for non-queried intersections, and the zorder assigned to
        a query when the class was instantiated otherwise
        :param query: frozenset.
        :return: zorder as int.
        """
        return self._query_to_zorder.get(query, 0)

    def _table_names_plot(self, ylim):
        """
        Plots the table names
        """
        axes = self._graph_store.names
        axes.set_ylim(ylim)
        for index, name in enumerate(self._data.sizes.keys()):
            self._row_map[name] = (index + 1)
            axes.text(
                x=1,
                y=self._y_values[index],
                s=name,
                fontsize=14,
                clip_on=True,
                va='center',
                ha='right',
                transform=axes.transData,
                family='monospace'
            )

        axes.axis('off')

    def _base_sets_plot(self):
        """
        Plots horizontal bar plot for base set sizes.

        :return: Axes.
        """
        height = .6
        plot = self._graph_store.base_set_size
        plot.invert_xaxis()

        bar_bottoms = self._y_values - (height / 2)
        plot.barh(
            bar_bottoms,
            [
                self._data.sizes[key]
                for key in self._data.sizes
            ],
            height=height,
            color=self._colours.greys[1]
        )

        plot.ticklabel_format(style='sci', axes='x', scilimits=(0, 4))

        self._strip_axes(plot, keep_spines=['bottom'], keep_ticklabels=['bottom'])

        plot.set_ylim(((height / 2), plot.get_ylim()[1] + (height / 2)))
        xlim = plot.get_xlim()

        gap = (max(xlim) / self.GAP_DIVISOR) * self.GAP_MULTIPLIER
        plot.set_xlim(xlim[0] + gap, xlim[1] - gap)
        xlim = plot.get_xlim()

        plot.spines['bottom'].set_bounds(xlim[0], xlim[1])
        plot.set_xlabel("Set size", fontweight='bold', fontsize=13)

        return plot.get_ylim()

    @staticmethod
    def _strip_axes(plot_axes, keep_spines=None, keep_ticklabels=None):
        """
        Removes spines and tick labels from ax, except those specified by the user.

        :param plot_axes: Plot axes on which to operate.
        :param keep_spines: Names of spines to keep.
        :param keep_ticklabels: Names of tick labels to keep.

        Possible names are 'left'|'right'|'top'|'bottom'.
        """
        tick_params_dict = {'which': 'both',
                            'bottom': 'off',
                            'top': 'off',
                            'left': 'off',
                            'right': 'off',
                            'labelbottom': 'off',
                            'labeltop': 'off',
                            'labelleft': 'off',
                            'labelright': 'off'}
        if keep_ticklabels is None:
            keep_ticklabels = []
        if keep_spines is None:
            keep_spines = []
        lab_keys = [(k, "".join(["label", k])) for k in keep_ticklabels]
        for k in lab_keys:
            tick_params_dict[k[0]] = 'on'
            tick_params_dict[k[1]] = 'on'
        plot_axes.tick_params(**tick_params_dict)
        for sname, spine in plot_axes.spines.items():
            if sname not in keep_spines:
                spine.set_visible(False)

    def _inters_sizes_plot(self):
        """
        Plots bar plot for intersection sizes.
        :param ordered_in_sets: array of tuples. Each tuple represents an intersection. The array is sorted according
        to the user's directives

        :param inters_sizes: array of ints. Sorted, likewise.

        :return: Axes
        """
        width = .5
        plot = self._graph_store.intersection_bars

        inters_sizes = [item.size for item in self._data.results]
        self._strip_axes(plot, keep_spines=['left'], keep_ticklabels=['left'])

        bar_bottom_left = self._x_values - width / 2
        bar_colors = [
            self._color_for_query(
                frozenset(item.in_sets)
            ) for item in self._data.results
        ]

        plot.bar(
            bar_bottom_left,
            inters_sizes,
            width=width,
            color=bar_colors,
            linewidth=0
        )

        ylim = plot.get_ylim()
        label_vertical_gap = (ylim[1] - ylim[0]) / 60

        for pos_x, pos_y in zip(self._x_values, inters_sizes):
            plot.text(
                pos_x,
                pos_y + label_vertical_gap,
                "%.2g" % pos_y,
                rotation=90,
                ha='center',
                va='bottom'
            )

        plot.ticklabel_format(style='sci', axes='y', scilimits=(0, 4))

        gap = max(ylim) / self.GAP_DIVISOR * self.GAP_MULTIPLIER
        plot.set_ylim(ylim[0] - gap, ylim[1] + gap)
        ylim = plot.get_ylim()
        plot.spines['left'].set_bounds(ylim[0], ylim[1])

        plot.yaxis.grid(True, lw=.25, color='grey', ls=':')
        plot.set_axisbelow(True)
        plot.set_ylabel("Intersection size", labelpad=6, fontweight='bold', fontsize=13)

        return plot.get_xlim()

    def _inters_matrix(self, xlims, ylims):
        """
        Plots intersection matrix.

        :param ordered_in_sets: Array of tuples representing sets included in an intersection. Sorted according to
        the user's directives.

        :param ordered_out_sets: Array of tuples representing sets excluded from an intersection. Sorted likewise.

        :param xlims: tuple. x limits for the intersection matrix plot.

        :param ylims: tuple. y limits for the intersection matrix plot.

        :param set_row_map: dict. Maps data frames (base sets) names to a row of the intersection matrix

        :return: Axes
        """
        plot = self._graph_store.intersection_matrix
        plot.set_xlim(xlims)
        plot.set_ylim(ylims)

        row_width = self._x_values[1] - self._x_values[0] if len(self._x_values) > 1 else self._x_values[0]
        self._strip_axes(plot)
        background = plt.cm.Greys([.09])[0]

        for row, y_pos in enumerate(self._y_values):
            if row % 2 == 0:
                plot.add_patch(
                    Rectangle(
                        (xlims[0], y_pos - row_width / 2),
                        height=row_width,
                        width=xlims[1],
                        color=background, zorder=0
                    )
                )

        ordered_in_sets = [item.in_sets for item in self._data.results]
        ordered_out_sets = [item.out_sets for item in self._data.results]

        for col_num, (in_sets, out_sets) in enumerate(zip(ordered_in_sets, ordered_out_sets)):
            in_y = [self._row_map[s] for s in in_sets]
            out_y = [self._row_map[s] for s in out_sets]

            plot.scatter(
                np.repeat(self._x_values[col_num], len(in_y)),
                in_y,
                color=np.tile(
                    self._color_for_query(frozenset(in_sets)),
                    (len(in_y), 1)
                ),
                s=300
            )
            plot.scatter(
                np.repeat(self._x_values[col_num], len(out_y)),
                out_y,
                color=self._colours.greys[0], s=300
            )
            plot.vlines(
                self._x_values[col_num],
                min(in_y),
                max(in_y),
                lw=3.5,
                color=self._color_for_query(frozenset(in_sets))
            )

    def _plot_additional(self):
        """ Wrapper method for additional plots """
        self._graph_store.additional = []

        for index, graph_settings in enumerate(self._additional_plots):
            plot_kind = graph_settings.pop('kind')
            data_vars = graph_settings.pop('data_quantities')
            graph_properties = graph_settings.get('graph_properties', {})

            plot = self._additional_plot(
                index,
                plot_kind,
                graph_properties,
                labels=data_vars
            )
            self._graph_store.additional.append(plot)

    def _additional_plot(self, ax_index, kind, graph_args, *, labels=None):
        """
        Scatter plot (for additional plots).

        :param ax_index: int. Index for the relevant axes (additional plots' axes are stored in a list)

        :param data_values: list of dictionary. Each dictionary is like {'x':data_for_x, 'y':data_for_y,
        'in_sets':tuple}, where the tuple represents the intersection the data for x and y belongs to.

        :param plot_kwargs: kwargs accepted by matplotlib scatter

        :param labels: dictionary. {'x':'x_label', 'y':'y_label'}

        :return: Axes
        """
        # pylint: disable=too-many-locals
        # Original function - needs many params

        plot = self.additional_plots_axes[ax_index]
        plot_method = getattr(plot, kind)

        for key, value in self._standard_graph_settings.get(kind, {}).items():
            graph_args.setdefault(key, value)

        plot_method = partial(plot_method, **graph_args)

        ylim, xlim = [np.inf, -np.inf], [np.inf, -np.inf]
        for result in self._data.results:
            highlight = frozenset(result.in_sets)
            values = result.additional_plot_data(**labels)
            plot_method(
                color=self._color_for_query(highlight),
                zorder=self._zorder_for_query(highlight),
                **values
            )
            new_xlim, new_ylim = plot.get_xlim(), plot.get_ylim()
            for old, new in zip([xlim, ylim], [new_xlim, new_ylim]):
                old[0] = new[0] if old[0] > new[0] else old[0]
                old[1] = new[1] if old[1] < new[1] else old[1]

        plot.ticklabel_format(style='sci', axes='y', scilimits=(0, 4))

        self._strip_axes(plot, keep_spines=['bottom', 'left'], keep_ticklabels=['bottom', 'left'])
        gap_y = max(ylim) / self.GAP_DIVISOR * self.GAP_MULTIPLIER
        gap_x = max(xlim) / self.GAP_DIVISOR * self.GAP_MULTIPLIER

        plot.set_ylim(ylim[0] - gap_y, ylim[1] + gap_y)
        plot.set_xlim(xlim[0] - gap_x, xlim[1] + gap_x)
        ylim, xlim = plot.get_ylim(), plot.get_xlim()
        plot.spines['left'].set_bounds(ylim[0], ylim[1])
        plot.spines['bottom'].set_bounds(xlim[0], xlim[1])

        for index, text in labels.items():
            # pylint: disable=expression-not-assigned
            # expression wraps a function call. Don't need to assign the output
            getattr(plot, 'set_%slabel' % index)(
                text,
                labelpad=3,
                fontweight='bold',
                fontsize=13
            ) if index in ['x', 'y'] else None
        return plot

    def _setup_colours(self):
        """ Setup the colour scheme """
        self._colours.greys = plt.cm.Greys([.22, .8])
        self._colours.standard = dict(
            zip(
                [frozenset(q) for q in self._highlight],
                plt.cm.rainbow(np.linspace(.01, .99, len(self._highlight)))
            )
        )

    def _setup_zorder(self):
        """ Sets up the index ordering (z-order) of highlighting """
        self._query_to_zorder = dict(
            zip(
                [frozenset(q) for q in self._highlight],
                np.arange(len(self._highlight)) + 1
            )
        )
