"""
Unit tests for the Visualisation module
"""

from unittest import TestCase
from ddt import ddt, data, unpack

import matplotlib
from pyupset.resources import FilterConfig
from pyupset.resources import SortMethods
from pyupset.resources import DataExtractor
from pyupset.visualisation import UpSetPlot

from tests.pyupset.test_resources import DataProvider
from pyupset import plot

@ddt
class TestUpsetPlot(TestCase):
    """
    Tests for the various methods of drawing UpSet diagrams
    """
    # pylint: disable=invalid-name
    # Test names are always long to be descriptive of the test they are doing.

    def test_upsetplot_raises_runtime_error(self):
        """ calls to UpSetPlot.__init__ will raise a runtime error if data_extractor is not DataExtractor class """
        dataframe = DataProvider.mock_dataframe()
        with self.assertRaises(RuntimeError):
            UpSetPlot(dataframe)

    @data(
        DataProvider.mock_dataframe(),
        DataProvider.mock_mixed_dictionary()
    )
    def test_init_no_query_no_additional(self, mock_dataframes):
        """ Basic test for initialisation """
        mock_unique = ['name']
        extractor = DataExtractor(unique_keys=mock_unique, filter_config=DataProvider.mock_filter())
        # pylint: disable=expression-not-assigned
        # We don't require the value of this list as
        # we're purely interested in list-comprehension
        # to make things simple.
        [extractor.append(name, mock_dataframes[name]) for name in mock_dataframes]

        upset = UpSetPlot(extractor)
        # pylint: disable=protected-access
        # Need to access protected properties to test for existance
        self.assertIsInstance(upset._graph_store.figure, matplotlib.figure.Figure)
        for key in upset._graph_store._options:
            if key == '_figure':
                continue
            elif key == '_additional':
                self.assertIsNone(upset._graph_store._options[key])
            else:
                self.assertEqual(
                    str(type(upset._graph_store._options[key])), "<class 'matplotlib.axes._subplots.AxesSubplot'>"
                )

        self.assertIsInstance(upset._grid_spec.main_gs, matplotlib.gridspec.GridSpec)
        self.assertIsInstance(upset._grid_spec.top_gs, matplotlib.gridspec.SubplotSpec)
        self.assertIsNone(upset._grid_spec.bottom_gs)

    @data(
        DataProvider.mock_dataframe(),
        DataProvider.mock_mixed_dictionary()
    )
    def test_init_with_additional_plots(self, mock_dataframes):
        """
        Tests for the various methods of drawing UpSet diagrams
        """
        mock_unique = ['name']
        extractor = DataExtractor(unique_keys=mock_unique, filter_config=DataProvider.mock_filter())
        # pylint: disable=expression-not-assigned
        # We don't require the value of this list as
        # we're purely interested in list-comprehension
        # to make things simple.
        [extractor.append(name, mock_dataframes[name]) for name in mock_dataframes]
        additional_plots = [
            {
                'kind':'scatter',
                'data_quantities': {
                    'x':'names',
                    'y':'values'
                }
            },
            {
                'kind': 'hist',
                'data_quantities': {
                    'x': 'values'
                }
            }
        ]
        extractor.build()
        upset = UpSetPlot(extractor, additional_plots=additional_plots)
        # pylint: disable=protected-access
        # Need to access protected properties to test for existance
        self.assertIsInstance(upset._graph_store.figure, matplotlib.figure.Figure)
        for key in upset._graph_store._options:
            if key == '_figure':
                continue
            elif key == '_additional':
                self.assertIsNone(upset._graph_store._options[key])
            else:
                self.assertEqual(
                    str(type(upset._graph_store._options[key])), "<class 'matplotlib.axes._subplots.AxesSubplot'>"
                )

        self.assertIsInstance(upset._grid_spec.main_gs, matplotlib.gridspec.GridSpec)
        self.assertIsInstance(upset._grid_spec.top_gs, matplotlib.gridspec.SubplotSpec)
        self.assertIsInstance(upset._grid_spec.bottom_gs, matplotlib.gridspec.SubplotSpec)

    @data(
        [DataProvider.mock_dataframe(), ['name'], 'tests/generated/dataframe.png'],
        [DataProvider.mock_mixed_dictionary(), ['name'], 'tests/generated/mixed_dictionary.png'],
        [DataProvider.pickled(), ['title'], 'tests/generated/pickled.png']

    )
    @unpack
    def test_plot(self, mock_dataframe, mock_unique, filename):
        """ Test the plot method """
        extractor = DataExtractor(unique_keys=mock_unique, filter_config=DataProvider.mock_filter())
        # pylint: disable=expression-not-assigned
        # We don't require the value of this list as
        # we're purely interested in list-comprehension
        # to make things simple.
        [extractor.append(name, mock_dataframe[name]) for name in mock_dataframe]
        extractor.build()
        upset = UpSetPlot(extractor)
        upset.plot()
        upset.save(filename)

    def test_degree_size_bounds(self):
        """ Test plotting with filter by degree and limiting size bounds """
        dataframe = DataProvider.pickled()
        filter_config = FilterConfig()
        filter_config.sort_by = SortMethods.DEGREE
        filter_config.size_bounds = (20, 400)
        filter_config.reverse = False
        extractor = DataExtractor(['title'], filter_config)

        # pylint: disable=expression-not-assigned
        # We don't require the value of this list as
        # we're purely interested in list-comprehension
        # to make things simple.
        [extractor.append(name, dataframe[name]) for name in dataframe]
        extractor.build()
        upset = UpSetPlot(extractor)
        upset.plot()
        upset.save('tests/generated/degree_of_pickled.png')

    def test_highlight(self):
        """ Validate highlighting """
        highlight = [('adventure', 'action'), ('romance', 'war')]
        dataframe = DataProvider.pickled()
        extractor = DataExtractor(unique_keys=['title'], filter_config=DataProvider.mock_filter())
        # pylint: disable=expression-not-assigned
        # We don't require the value of this list as
        # we're purely interested in list-comprehension
        # to make things simple.
        [extractor.append(name, dataframe[name]) for name in dataframe]
        extractor.build()
        upset = UpSetPlot(extractor, highlight=highlight)
        upset.plot()
        upset.save('tests/generated/pickle_highlighted.png')

    def test_additional(self):
        """ Validate  plotting using additional arguments """
        highlight = [('adventure', 'action'), ('romance', 'war')]
        additional_plots = [
            {
                'kind': 'scatter',
                'data_quantities':{
                    'x': 'views',
                    'y':'rating_std'
                }
            },
            {
                'kind': 'hist',
                'data_quantities': {
                    'x': 'views'
                }
            }
        ]
        dataframe = DataProvider.pickled()
        extractor = DataExtractor(unique_keys=['title'], filter_config=DataProvider.mock_filter())
        # pylint: disable=expression-not-assigned
        # We don't require the value of this list as
        # we're purely interested in list-comprehension
        # to make things simple.
        [extractor.append(name, dataframe[name]) for name in dataframe]
        extractor.build()
        upset = UpSetPlot(extractor, highlight=highlight, additional_plots=additional_plots)
        upset.plot()
        upset.save('tests/generated/additional_pickle.png')

    def test_additional_with_extra_arguments(self):
        """ Validate  plotting using enhanced additional arguments """
        highlight = [('adventure', 'action'), ('romance', 'war')]
        additional_plots = [
            {
                'kind':'scatter',
                'data_quantities':{
                    'x': 'views',
                    'y': 'rating_std'
                },
                'graph_properties':{
                    'alpha':.8,
                    'lw': .4,
                    'edgecolor': 'w',
                    's':50
                }
            },
            {
                'kind':'hist',
                'data_quantities':{'x': 'views'},
                'graph_properties':{'bins': 50}
            }
        ]
        dataframe = DataProvider.pickled()
        extractor = DataExtractor(unique_keys=['title'], filter_config=DataProvider.mock_filter())
        # pylint: disable=expression-not-assigned
        # We don't require the value of this list as
        # we're purely interested in list-comprehension
        # to make things simple.
        [extractor.append(name, dataframe[name]) for name in dataframe]
        extractor.build()
        upset = UpSetPlot(extractor, highlight=highlight, additional_plots=additional_plots)
        upset.plot()
        upset.save('tests/generated/extra_additional_pickle.png')

    def test_methods_plot_function(self):
        """ Validate the call via pyupset.plot """
        dataframes = DataProvider.mock_mixed_dictionary()
        results = plot(dataframes)
        keys = ['figure', 'intersection_bars', 'intersection_matrix', 'base_set_size', 'names']
        self.assertEqual(sorted(keys), sorted(results.keys()))

    def test_methods_plot_function_with_additional(self):
        """ Validate the call via pyupset.plot """
        dataframes = DataProvider.pickled()
        additional_plots = [
            {
                'kind':'scatter',
                'data_quantities':{
                    'x': 'views',
                    'y': 'rating_std'
                },
                'graph_properties':{
                    'alpha':.8,
                    'lw': .4,
                    'edgecolor': 'w',
                    's':50
                }
            },
            {
                'kind':'hist',
                'data_quantities':{'x': 'views'},
                'graph_properties':{'bins': 50}
            }
        ]
        results = plot(dataframes, unique_keys=['title'], additional_plots=additional_plots)
        keys = ['figure', 'intersection_bars', 'intersection_matrix', 'base_set_size', 'names', 'additional']
        self.assertEqual(sorted(keys), sorted(results.keys()))
