"""
Unit tests for the resources module
"""

from collections import namedtuple
from pickle import load
from unittest import TestCase
from mock import patch

from ddt import ddt, data, unpack
import numpy as np
import pandas as pd
from pyupset.resources import FilterConfig
from pyupset.resources import SortMethods
from pyupset.resources import ExtractedData
from pyupset.resources import DataExtractor

class DataProvider(object):
    """ Data for use within tests """
    @staticmethod
    def mock_dataframe():
        """ Provides a dictionary of dataframes for use within tests """

        dictionary = {
            'dataframe_one': None,
            'dataframe_two': None,
            'dataframe_three': None
        }

        columns = ['name', 'value']
        dictionary['dataframe_one'] = pd.DataFrame(
            {
                'Name': ['aaa', 'bbb', 'ccc', 'fyx', 'yxy'],
                'value': [24, 28, 36, 124, 256]
            }
        )
        dictionary['dataframe_one'].columns = columns

        dictionary['dataframe_two'] = pd.DataFrame(
            {
                'name': ['bcb', 'bac', 'ccc', 'fyx', 'yxz'],
                'value': [4, 21, 44, 14, 56]
            }
        )
        dictionary['dataframe_two'].columns = columns

        dictionary['dataframe_three'] = pd.DataFrame(
            {
                'name': ['aaa', 'bac', 'bbb', 'fyx', 'zzz'],
                'value': [4, 21, 28, 14, 59]
            }
        )
        dictionary['dataframe_three'].columns = columns

        return dictionary

    @staticmethod
    def mock_mixed_dictionary():
        """ Provides multiple dictionaries of dataframes for validating combinations """

        # mixed column count
        dictionary = {
            'dataframe_one': None,
            'dataframe_two': None,
            'dataframe_three': None
        }
        dictionary['dataframe_one'] = pd.DataFrame(
            {
                'name': ['aaa', 'bbb', 'ccc', 'fyx', 'yxy'],
                'value': [24, 28, 36, 124, 256],
                'city': ['London', 'Manchester', 'Leeds', 'York', 'Birmingham']
            }
        )

        dictionary['dataframe_two'] = pd.DataFrame(
            {
                'name': ['bcb', 'bac', 'ccc', 'fyx', 'yxz'],
                'value': [4, 21, 44, 14, 56]
            }
        )

        dictionary['dataframe_three'] = pd.DataFrame(
            {
                'name': ['aaa', 'bac', 'bbb', 'fyx', 'zzz', 'hhh'],
                'value': [4, 21, 28, 14, 59, 279],
                'city': ['Madrid', 'Barcelona', 'Seville', 'Granada', 'Bilbao', 'Valencia']
            }
        )
        return dictionary

    @staticmethod
    def pickled():
        """ Load the original PyUpset test data """
        with open('tests/pyupset/data/test_data_dict.pckl', 'rb') as pickled:
            return load(pickled)

    @staticmethod
    def mock_merged_dataframe():
        """ Test result frame """
        dataframe = {
            'name': ['aaa', 'bbb', 'ccc', 'fyx', 'yxy', 'bac', 'zzz', 'bcb', 'yxz'],
            'value_dataframe_one': [24., 28., 36., 124., 256., np.nan, np.nan, np.nan, np.nan],
            'value_dataframe_three': [4., 28., np.nan, 14., np.nan, 21., 59., np.nan, np.nan],
            'value_dataframe_two': [np.nan, np.nan, 44., 14., np.nan, 21., np.nan, 4., 56.]
        }
        return pd.DataFrame(dataframe)

    @staticmethod
    def mock_filter():
        """ A filter object for use within tests """
        config = FilterConfig()
        config.sort_by = 'SIZE'
        config.degree_bounds = (1, np.inf)
        config.size_bounds = (0, np.inf)
        config.reverse = True
        return config

@ddt
class TestFilterConfig(TestCase):
    """ Tests for pyupset.resources.FilterConfig """

    # pylint: disable=invalid-name
    # Method names should be descriptive of the task they are doing
    # for tests, this can often mean long names

    @data(
        'TestAttribute',
        '_test_attribute',
        'test_attribute'
    )
    def test_getattribute_raises_exception_if_attribute_is_invalid(self, attribute):
        """ Attributes not accepted by FilterConfig raise AttributeError """
        with self.assertRaises(AttributeError):
            getattr(FilterConfig(), attribute)

    @data(
        ('size_bounds', (1, 2), (3, 4)),
        ('sort_by', 'DEGREE', 'SIZE'),
        ('degree_bounds', (8, 18), (9, 34))
    )
    @unpack
    def test_setattribute_raises_exception_if_attribute_already_set(
            self,
            attribute,
            value,
            replacement_value
    ):
        """ Attributes are immutable once set and raise AttributeError on re-assignment """

        config = FilterConfig()

        options = {'_size_bounds': None, '_sort_by': None, '_degree_bounds': None, '_reverse': None}
        # pylint: disable=protected-access
        self.assertEqual(config._options, options)
        setattr(config, attribute, value)

        with self.assertRaises(RuntimeError):
            setattr(config, attribute, replacement_value)

    def test_setattribute_raises_exception_if_attribute_has_lambda_mapping(self):
        """ Attributes with lambda mappings may raise an AttributeError """
        config = FilterConfig()
        with self.assertRaises(AttributeError):
            config.sort_by = 1

    def test_setattributes(self):
        """ Happy path test """
        config = FilterConfig()
        config.sort_by = 'SIZE'
        config.degree_bounds = (123, 456)
        config.size_bounds = (1, np.inf)

        self.assertEqual(SortMethods.SIZE, config.sort_by)
        self.assertEqual((123, 456), config.degree_bounds)
        self.assertEqual((1, np.inf), config.size_bounds)

@ddt
class TestExtractedData(TestCase):
    """ Tests Data extraction via calls to DataExtractor class """

    # pylint: disable=invalid-name
    # Method names should be descriptive of the task they are doing
    # for tests, this can often mean long names

    @data(
        [DataProvider.mock_dataframe(), 7, [5, 5, 5], 3, 7],
        [DataProvider.mock_mixed_dictionary(), 7, [5, 5, 6], 3, 7]
    )
    @unpack
    def test_extract(self, mock_dataframes, mock_count, mock_sizes, mock_rows, mock_cols):
        """ Validate the data filter """
        # pylint: disable=too-many-arguments
        # Mocked data - arguments required
        mock_unique = ['name']
        base_sets = DataExtractor(unique_keys=mock_unique, filter_config=DataProvider.mock_filter())
        # pylint: disable=expression-not-assigned
        # We don't require the value of this list as
        # we're purely interested in list-comprehension
        # to make things simple.
        [base_sets.append(name, mock_dataframes[name]) for name in mock_dataframes]

        self.assertIsInstance(base_sets.merge, pd.DataFrame)
        self.assertEqual(mock_count, len(base_sets.extract()))

        sizes = []
        for key in base_sets.sizes:
            sizes.append(base_sets.sizes[key])
        self.assertEqual(sorted(mock_sizes), sorted(sizes))

        self.assertEqual(mock_rows, base_sets.rows)
        self.assertEqual(mock_cols, base_sets.columns)
        self.assertEqual(mock_count, base_sets.columns)

    def test_append_raises_exception_if_key_has_already_been_set(self):
        """ Validate the data filter """
        mock_dataframes = DataProvider.mock_dataframe()
        mock_unique = ['name']
        base_sets = DataExtractor(unique_keys=mock_unique)
        # pylint: disable=expression-not-assigned
        # We don't require the value of this list as
        # we're purely interested in list-comprehension
        # to make things simple.
        [base_sets.append(name, mock_dataframes[name]) for name in mock_dataframes]
        with self.assertRaises(AttributeError):
            base_sets.append('dataframe_one', {})

    def test_extract_with_sub_query(self):
        """ Sub queries repeat for replacements """
        query = '{value} > 30'
        comparison_query = '(value_dataframe_three > 30) or (value_dataframe_one > 30) or (value_dataframe_two > 30)'
        mock_unique = ['name']

        base_sets = DataExtractor(unique_keys=mock_unique)
        base_sets.names = DataProvider.mock_dataframe().keys()
        base_sets.merge = DataProvider.mock_merged_dataframe()
        base_sets.sub_query = query

        self.assertEqual(len(comparison_query), len(base_sets.sub_query))
        self.assertTrue(base_sets.merge.equals(DataProvider.mock_merged_dataframe()))
        self.assertEqual(7, len(base_sets.extract()))

    def test_extract_with_sub_query_skips_missing_columns(self):
        """ Sub queries skip columns which don't exist """
        query = '{city} = \'London\''
        comparison_query = '(city_dataframe_one = \'London\') or (city_dataframe_three = \'London\')'
        mock_unique = ['name']
        mock_dataframes = DataProvider.mock_mixed_dictionary()

        base_sets = DataExtractor(unique_keys=mock_unique)
        # pylint: disable=expression-not-assigned
        # We don't require the value of this list as
        # we're purely interested in list-comprehension
        # to make things simple.
        [base_sets.append(name, mock_dataframes[name]) for name in mock_dataframes]
        base_sets.sub_query = query
        self.assertEqual(len(comparison_query), len(base_sets.sub_query))

    def test_sub_query_with_exact_columns(self):
        """ Sub queries can contain merged names """
        query = 'value_dataframe_one > 30'
        comparison_query = 'value_dataframe_one > 30'
        mock_unique = ['name']

        base_sets = DataExtractor(unique_keys=mock_unique)
        base_sets.names = DataProvider.mock_dataframe().keys()
        base_sets.merge = DataProvider.mock_merged_dataframe()
        base_sets.sub_query = query
        self.assertEqual(comparison_query, base_sets.sub_query)

    def test_parse_sub_query_raises_exception_if_no_dataframe_names_have_been_provided(self):
        """ If no database names have been provided, an error will be raised """
        query = 'value_dataframe_one > 30'
        mock_unique = ['name']

        base_sets = DataExtractor(unique_keys=mock_unique)
        base_sets.merge = DataProvider.mock_merged_dataframe()
        with self.assertRaises(RuntimeError):
            base_sets.sub_query = query

    def test_set_names_raises_runtime_error_if_names_already_provided(self):
        """ You cannot provide names if you've already provided seperate dataframes """
        mock_dataframes = DataProvider.mock_dataframe()
        mock_unique = ['name']
        base_sets = DataExtractor(unique_keys=mock_unique)
        # pylint: disable=expression-not-assigned
        # We don't require the value of this list as
        # we're purely interested in list-comprehension
        # to make things simple.
        [base_sets.append(name, mock_dataframes[name]) for name in mock_dataframes]
        with self.assertRaises(RuntimeError):
            base_sets.names = DataProvider.mock_dataframe().keys()

    def test_get_columns_raises_error_if_called_before_extract(self):
        """ Calling columns before extracting the data should raise a RuntimeError """
        mock_unique = ['name']
        base_sets = DataExtractor(unique_keys=mock_unique)
        with self.assertRaises(RuntimeError):
            self.assertIsNone(base_sets.columns)

    @data(
        [1, 2, 3, 4],
        1234,
        '1234',
        True
    )
    def test_merge_setter_only_accepts_pandas_dataframe(self, merge_type):
        """ Only pandas.DataFrame objects can be passed to the merge setter """
        mock_unique = ['name']
        base_sets = DataExtractor(unique_keys=mock_unique)
        with self.assertRaises(TypeError):
            base_sets.merge = merge_type

    def test_merge_cleans_up_and_raises_memory_error(self):
        """ If memory errors are raised within Pandas, merge should clean up after it's self """
        mock_dataframes = DataProvider.mock_dataframe()
        mock_unique = ['name']
        base_sets = DataExtractor(unique_keys=mock_unique)
        # pylint: disable=expression-not-assigned
        # We don't require the value of this list as
        # we're purely interested in list-comprehension
        # to make things simple.
        [base_sets.append(name, mock_dataframes[name]) for name in mock_dataframes]
        with patch('pandas.DataFrame.merge') as mock_merge:
            mock_merge.side_effect = MemoryError('Out of memory')
            with self.assertRaises(MemoryError):
                results = base_sets.merge
                # pylint: disable=protected-access
                # Need to access _merge to validate cleanup. Calling `merge` would fail.
                self.assertIsNone(base_sets._merge)
                self.assertIsNone(results)

    def test_comparisons_by_size(self):
        """ Test ExtractedData comparison methods using SortMethods.SIZE """
        results = namedtuple('Results', 'index')
        results_one = results(index=[1, 2, 3, 4])
        results_two = results(index=[1, 2, 3, 4, 5, 6])

        filter_config = FilterConfig()
        filter_config.sort_by = SortMethods.SIZE

        dataset_one = ExtractedData(
            ('test', 'test_one'),
            ('no_test', 'no_test_one'),
            results_one,
            filter_config
        )
        dataset_two = ExtractedData(
            ('test_two', 'test_one'),
            ('no_test_two', 'no_test_one'),
            results_two,
            filter_config
        )
        dataset_three = ExtractedData(
            ('test_two', 'test_one'),
            ('no_test_two', 'no_test_one'),
            results_one,
            filter_config
        )

        self.assertTrue(dataset_one < dataset_two)
        self.assertTrue(dataset_one <= dataset_two)
        self.assertTrue(dataset_two > dataset_one)
        self.assertTrue(dataset_two >= dataset_one)
        self.assertEqual(dataset_one, dataset_three)
        self.assertNotEqual(dataset_one, dataset_two)

    def test_comparisons_by_degree(self):
        """ Test ExtractedData comparison methods using SortMethods.DEGREE """
        results = namedtuple('Results', 'index')
        results_one = results(index=[1, 2, 3, 4])
        results_two = results(index=[1, 2, 3, 4, 5, 6])

        filter_config = FilterConfig()
        filter_config.sort_by = SortMethods.DEGREE

        dataset_one = ExtractedData(
            ('test', 'test_one'),
            ('no_test', 'no_test_one'),
            results_one,
            filter_config
        )
        dataset_two = ExtractedData(
            ('test_two', 'test_one', 'test_three'),
            ('no_test_two', 'no_test_one'),
            results_two,
            filter_config
        )
        dataset_three = ExtractedData(
            ('test_two', 'test_one'),
            ('no_test_two', 'no_test_one'),
            results_one,
            filter_config
        )

        self.assertTrue(dataset_one < dataset_two)
        self.assertTrue(dataset_one <= dataset_two)
        self.assertTrue(dataset_two > dataset_one)
        self.assertTrue(dataset_two >= dataset_one)
        self.assertEqual(dataset_one, dataset_three)
        self.assertNotEqual(dataset_one, dataset_two)
