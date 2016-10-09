"""
Resource classes utilised by the PyUpset module

:author: Martin Proffitt <mproffitt@jitsc.co.uk>
:since: v0.1.2

"""
from collections import OrderedDict
from enum import Enum
from itertools import chain
from itertools import combinations
import numpy as np
import pandas as pd

class SortMethods(Enum):
    """
    Enum definition of the available sort methods
    """
    # pylint: disable=too-few-public-methods
    # Enumeration types do not require any methods

    SIZE = 'SIZE'
    DEGREE = 'DEGREE'

class Immutable(object):
    """
    Parent Class for creating immutable objects.

    Child classes should then provide two attributes:

        _options        A dictionary of options which should be immutable
        _options_setter A dictionary of lambda functionality mapping onto options [optional]
    """
    # pylint: disable=too-few-public-methods
    # Immutable instance parent defines attribute access methods

    _option_setter = {}
    _options = {}

    def __getattr__(self, attribute):
        """
        Get the value of a requested attribute

        @param attribute string

        @return attribute value

        @throws AttributeError if the requested attribute does not exist.
        """
        try:
            if object.__getattribute__(self, attribute) is not None:
                return object.__getattribute__(self, attribute)
        except AttributeError:
            pass

        if not attribute.startswith('_'):
            attribute = '_' + attribute

        if attribute not in self._options.keys():
            raise AttributeError(
                '\'{attribute}\' not a valid attribute for \'{klass}\''.format(
                    attribute=attribute,
                    klass=type(self))
            )

        if self._options[attribute] is None and hasattr(self, '_defaults'):
            try:
                return self._defaults[attribute]
            except AttributeError:
                pass
        return self._options[attribute]

    def __setattr__(self, attribute, value):
        """
        Set the value of the requested attribute

        @param attribute string
        @param value     string
        """
        try:
            if object.__getattribute__(self, attribute) is not None:
                return object.__setattr__(self, attribute, value)
        except AttributeError:
            pass

        if not attribute.startswith('_'):
            attribute = '_' + attribute

        existing = getattr(self, attribute)
        default = None if not hasattr(self, '_defaults') else self._defaults[attribute]
        if getattr(self, attribute) is not None and existing != default:
            raise RuntimeError(
                'Cannot re-assign value of FilterConfig.{attribute} - is immutable'.format(
                    attribute=attribute
                )
            )
        try:
            if attribute in self._option_setter.keys():
                value = self._option_setter[attribute](value)
            self._options[attribute] = value
        except TypeError:
            raise AttributeError(
                'Invalid value \'{value}\' provided for attribute \'{attribute}\''.format(
                    value=value,
                    attribute=attribute
                )
            )

class Colours(Immutable):
    """ Colour store """
    # pylint: disable=too-few-public-methods
    # Colours is a struct type inhetiting Immutable.
    # Does not require public methods

    _options = {}

    def __init__(self):
        """ Create a new Colour store """
        self._options = {
            '_greys': None,
            '_standard': None
        }

class GraphStore(Immutable):
    """
    Graphs created within the visualisation will be stored here
    """
    # pylint: disable=too-few-public-methods
    # GraphStore is a struct type inhetiting Immutable.
    # Does not require public methods

    _options = {}
    def __init__(self):
        """
        Create an empty graph store
        """
        self._options = {
            '_figure': None,
            '_intersection_bars': None,
            '_intersection_matrix': None,
            '_base_set_size': None,
            '_names': None,
            '_additional': None
        }

class GridSpecStore(Immutable):
    """
    Grid based layout storage
    """
    # pylint: disable=too-few-public-methods
    # Filter config is a struct type inhetiting Immutable.
    # Does not require public methods

    _options = {}
    def __init__(self):
        """
        Create an empty GridSpec store
        """
        self._options = {
            '_main_gs': None,
            '_top_gs': None,
            '_bottom_gs': None
        }

class FilterConfig(Immutable):
    """
    Configuration for filtering the dataset

    The FilterConfig object is an immutable type allowing
    the parameters stored in _options dict to be set exactly
    once.

    The FilterConfig object has the following properties defined:

        * sort_by       string one of SIZE, DEGREE
        * size_bounds   tuple (x, y)
        * degree_bounds tuple (x, y)
        * reverse       bool If True, filter output will be sorted in
                             descending order instead of ascending [Default False]

    Used as a parameter to ExtractedData.filter called from the
        ExtractedData.filtered property.

    """
    # pylint: disable=too-few-public-methods
    # Filter config is a struct type inhetiting Immutable.
    # Does not require public methods

    _option_setter = {
        # pylint: disable=unnecessary-lambda,undefined-variable
        # This is required to hold FilterConfig until defined
        '_sort_by': lambda x: FilterConfig._sort(x)
    }

    _options = {}
    _defaults = {}

    def __init__(self):
        """
        Create an empty FilterConfig object
        """

        self._options = {
            '_size_bounds': None,
            '_degree_bounds': None,
            '_sort_by': None,
            '_reverse': None
        }

        self._defaults = {
            '_size_bounds':(0, np.inf),
            '_degree_bounds': (1, np.inf),
            '_sort_by': SortMethods.SIZE,
            '_reverse': False
        }

    @staticmethod
    def _sort(value):
        """ Validate the sort method against ``SortMethods`` """
        if isinstance(value, SortMethods):
            return value
        if hasattr(SortMethods, value):
            return getattr(SortMethods, value.upper())

class ExtractedData(Immutable):
    """
    Struct for storing data extracted via the DataExtractor class

    Struct properties

        in_sets: A list of frames / columns included in the result
        out_sets: A list of frames / columns excluded from the result
        size: Count of rows within the intersection (equivelant to pandas.DataFrame.count())
        results: The result of the query
    """
    # pylint: disable=too-few-public-methods
    # As an immutable struct class, we don't need to define any methods here.

    _option_setter = {}
    _options = {}

    def __init__(self, in_sets, out_sets, results, filter_config):
        self._options = {
            '_in_sets': in_sets,
            '_out_sets': out_sets,
            '_size': len(results.index),
            '_degree': len(in_sets),
            '_results': results,
            '_filter_config': filter_config
        }

    def additional_plot_data(self, x, y=None):
        """
        Prepares data for the additional plots

        :param string x_column:
        :param string y_column:

        :return dict:
        """
        # pylint: disable=invalid-name
        # x and y are set from configuration. Need to be left alone.
        values = {'x': self._plot_data(x)}
        if y is not None:
            values['y'] = self._plot_data(y)
        return values

    def _plot_data(self, column):
        """ Gets the data for a given plot as a flattened list """
        results = None
        for name in self.in_sets:
            column_name = '{column}_{frame}'.format(column=column, frame=name)
            frame = pd.DataFrame(self.results[column_name])
            frame.columns = [column]
            results = frame if results is None or results.empty else results.append(frame)
        results = list(results.dropna().values.tolist())
        return [value for item in results for value in item]

    def __gt__(self, other):
        if self._filter_config.sort_by == SortMethods.SIZE:
            return self.size > other.size
        return self.degree > other.degree

    def __ge__(self, other):
        if self._filter_config.sort_by == SortMethods.SIZE:
            return self.size >= other.size
        return self.degree >= other.degree

    def __lt__(self, other):
        if self._filter_config.sort_by == SortMethods.SIZE:
            return self.size < other.size
        return self.degree < other.degree

    def __le__(self, other):
        if self._filter_config.sort_by == SortMethods.SIZE:
            return self.size <= other.size
        return self.degree <= other.degree

    def __eq__(self, other):
        if self._filter_config.sort_by == SortMethods.SIZE:
            return self.size == other.size
        return self.degree == other.degree

    def __ne__(self, other):
        if self._filter_config.sort_by == SortMethods.SIZE:
            return self.size != other.size
        return self.degree != other.degree

class DataExtractor(object):
    """
    Store and operate on base-set information

    This class provides the core storage and query operation methods
    for generating graph data.

    The primary use of this class is to merge n* DataFrames then look for
    levels of intersection between them.

    Intersections are calculated as::

        SETS = A x B x ... Z
        intersection = len(SETS[x].value != NaN) for x in SETS

    See :meth:`DataExtractor.query` for a more complete example.

    :Example usage:
    Data for this example can be found in tests DataProvider class in tests/pyupset/test_resources.py

    ::

        from tests.pyupset.test_resources import TestExtractedData
        from pyupset.resources import DataExtractor

        base = DataExtractor(unique_keys=['name'])
        data = TestExtractedData.mock_dataframe()
        [base.append(name, data[name]) for name in data]
        results = base.build()

    Alternatively we may wish to specify a pre-built merge table::

        merge_table = TestExtractedData._mock_merged_dataframe()
        base = DataExtractor(unique_keys=['name'])
        base.merge = merge_table
        results = base.build()

    .. note:: If a merge table is provided, the ``append`` method has no effect.

    Sub Queries
    Sub-queries are also possible:

    Example::

        merge_table = TestExtractedData._mock_merged_dataframe()
        base = DataExtractor(unique_keys=['name'])
        base.merge = merge_table

        base.sub_query = 'value_dataframe_two > 30'
        results = base.build()

    See the :meth:`DataExtractor.sub_query` property for more
    information surrounding sub-queries.
    """

    # pylint: disable=too-many-instance-attributes
    # This is a fairly complex class although it only has one purpose.
    # Not splitting... For now.

    _frames = None
    _merge = None
    _unique_keys = None
    _non_unique_keys = None
    _filter_config = None
    _results = None
    _sub_query = None
    _pre_merge = False
    _sizes = None

    def __init__(self, unique_keys, filter_config=None):
        """
        Create a base set of data

        :param list unique_keys
        :param FilterConfig filter_config

        If ``filter_config`` is None, a default configuration will be provided sorting by degrees
        with the sizes and degrees bounds set to (0, np.inf)
        """
        if filter_config is None:
            filter_config = DataExtractor._get_default_filter()
        self._filter_config = filter_config
        self._unique_keys = unique_keys
        self._frames = {}
        self._results = None
        self._sizes = {}

    @property
    def ready(self):
        """
        :getter: Return true if the current object has results
        """
        return self._results is not None

    @property
    def results(self):
        """
        :getter: Return the current results set
        """
        return self._results

    @property
    def sub_query(self):
        """
        Pass additional queries to the merge

        :getter: Return the given sub-query
        :setter: Provide a sub-query to the data-extractor
        :type: string

        For a more detailed view of the syntax available for queries, see
        http://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.query.html

        When a sub-query is provided, specific ranges within the DataFrame can be targetted for the plot.
        Short queries can be blanket applied to all non-unique columns within the final merge table.

        For example, given the query string ``{value} > 30`` and the dataframe detailed in the
        :meth:`DataExtractor.query` method, the query would be re-written as::

            (value_dataframe_one > 30) or (value_dataframe_two > 30) or (value_dataframe_three > 30)

        This allows up-set plots to target very specific data-ranges within a larger DataFrame.

        If you wish to target the data in a single column only, you need to provide the exact column name as
        if appears within the merge. These columns are always formatted as ``[column name]_[dataframe name]``

        Column names to be replaced within the string should be wrapped in braces (`{}`) as in the sample above,
        """
        return self._sub_query

    @sub_query.setter
    def sub_query(self, query):
        """
        An optional query to append to the filter for each result-set

        :param query string
        """
        self._sub_query = self._parse_subquery(query)

    @property
    def merge(self):
        """
        Merge all dataframes into one large dataframe for querying

        :getter: pandas.DataFrame
        :setter: Set a pre-build dataframe
        :type: pandas.DataFrame

        :raises MemoryError: if the amount of memory required for the merge
                             is greater than the amount of memory available to
                             the system.

        In certain instances, we want to be able to pre-build the merge table
        and then assign that DataFrame into the library for graphing against.

        This is a key requirement for dataframes which may require a large amount
        of memory to process and otherwise cause issues within other parts of
        your application.

        .. warning:: Volatile. If the amount of memory required by the merge exceeds that available to the system, \
                it may cause the application to be killed without warning by the system kernel or cause the system \
                to become unstable. This is a built in limitation which at present cannot be avoided.

        If passing in a merge table built externally, you must also call the :meth:`DataExtractor.names` setter
        as these cannot be automatically determined.
        """
        if self._merge is not None:
            return self._merge

        try:
            keys = sorted(list(self._frames.keys()))
            for index, key in enumerate(keys):
                frame = self._frames[key]
                frame.columns = [
                    '{0}_{1}'.format(column, key)
                    if column not in self._unique_keys
                    else
                    column
                    for column in self._frames[key].columns
                ]
                if index == 0:
                    self._merge = frame
                else:
                    suffixes = (
                        '_{0}'.format(keys[index-1]),
                        '_{0}'.format(keys[index]),
                    )
                    self._merge = self._merge.merge(
                        frame,
                        on=self._unique_keys,
                        how='outer',
                        copy=False,
                        suffixes=suffixes
                    )
            return self._merge
        except MemoryError:
            # We want to clean up as best we can here to
            # relinquish resource back into the system
            del self._merge
            self._merge = None
            raise

    @merge.setter
    def merge(self, dataframe):
        """
        Allow the merge table to be set manually
        """
        if not isinstance(dataframe, pd.DataFrame):
            raise TypeError('dataframe must be a pandas.DataFrame object')
        self._merge = dataframe
        self._pre_merge = True

    @property
    def names(self):
        """
        A list of dataframe names given to the current object

        :getter: Return a list of names determined from _frames.keys()
        :setter: Sets the names of the dataframes used in the _merge table

        :raises RuntimeError: If you attempt to set names after individual dataframes have been provided.
        """
        return self._frames.keys()

    @names.setter
    def names(self, names):
        """ A list of dataframe names """
        if len(self._frames.keys()) > 0:
            raise RuntimeError('Dataframes have already been provided.')
        for name in names:
            self._frames[name] = {}

    @property
    def rows(self):
        """ Get the number of rows to write onto the graph """
        return len(self.names)

    @property
    def columns(self):
        """ Get the number of columns to write for the graph """
        if self._results is None:
            raise RuntimeError('Call to columns executed before call to build. No data to display')
        return len(self._results)

    @property
    def sizes(self):
        """
        Get the sizes of each given base set.

        If a merge table is provided, this method works out the unique size (excluding NaN) of each set used for the
        original merge
        """
        return OrderedDict(sorted(self._sizes.items(), key=lambda key: key[1], reverse=True))

    def append(self, name, dataframe):
        """
        Append a dataframe to the set of base sets

        :param name      string           A unique qualifying name for the dataframe
        :param dataframe pandas.DataFrame Dataframe to include in the graph

        :raises AssertionError: if a key with `name` has been set and is None
        :raises AttributeError: if a key with `name` has been set and is not None
        """
        try:
            assert self._frames[name] is not None
            raise AttributeError('A key with name \'{0}\' has already been set')
        except KeyError:
            self._frames[name] = dataframe

    def query(self, in_sets, out_sets, sub_query=None):
        """
        Build a query string to explicitely search for rows which match in_sets but where out_sets are NaN

        :param list in_sets:   A set of column names to explicitly include in the results
        :param list out_sets:  Column names to ensure are NaN
        :param str  sub_query: An optional query to filter the results further.

        :return: pandas.DataFrame The result of the query when executed against the merge DataFrame

        This method exploits a feature of numpy in which np.NaN is not equal to np.NaN to perform
        the comparison feature. Therefore, each column value is checked for equality against its-self to
        ensure out_sets are NaN and in-sets contain valid values.

        For example, Given the dataframe::

               name  value_dataframe_one  value_dataframe_three  value_dataframe_two
            0  aaa                 24.0                    4.0                  NaN
            1  bbb                 28.0                   28.0                  NaN
            2  ccc                 36.0                    NaN                 44.0
            3  fyx                124.0                   14.0                 14.0
            4  yxy                256.0                    NaN                  NaN
            5  bac                  NaN                   21.0                 21.0
            6  zzz                  NaN                   59.0                  NaN
            7  bcb                  NaN                    NaN                  4.0
            8  yxz                  NaN                    NaN                 56.0

        we would then write a query such that::

            query = "(value_dataframe_one != value_dataframe_one) & \\
                     (value_dataframe_two == value_dataframe_two) & \\
                     (value_dataframe_three != value_dataframe_three)"

        Which gives the result set::

              name  value_dataframe_one  value_dataframe_three  value_dataframe_two
            7  bcb                  NaN                    NaN                  4.0
            8  yxz                  NaN                    NaN                 56.0

        Sub Queries.

        A sub-query is a query to append to the primary filter enabling us to be
        even more specific with our results.

        For example, given the result set above, if we were only interested in rows
        in which the `value_dataframe_two` column is > 30, we would add this as a subquery
        such that::

            DataExtractor(...).query(in_sets, out_sets, subquery='value_dataframe_two > 30')

        would reveal::

              name  value_dataframe_one  value_dataframe_three  value_dataframe_two
            8  yxz                  NaN                    NaN                 56.0

        See :meth:`DataExtractor.sub_query` for examples of writing the sub-query string,
        """
        inclusive_columns = []
        for key in in_sets:
            for column in self._non_unique_keys:
                framed_column = '{column}_{frame}'.format(column=column, frame=key)
                if framed_column in self.merge.columns:
                    inclusive_columns.append(framed_column)
        inclusive_queries = ['({0} == {0})'.format(column) for column in inclusive_columns]

        exclusive_columns = []
        for key in out_sets:
            for column in self._non_unique_keys:
                framed_column = '{column}_{frame}'.format(column=column, frame=key)
                if framed_column in self.merge.columns:
                    exclusive_columns.append(framed_column)

        exclusive_queries = ['({0} != {0})'.format(column) for column in exclusive_columns]
        query = ' and '.join(inclusive_queries)
        if len(exclusive_queries) != 0:
            if len(inclusive_queries) != 0:
                query += ' and '
            query += ' and '.join(exclusive_queries)

        if sub_query is not None and sub_query != '':
            query += ' and ({query})'.format(query=self.sub_query)

        return self.merge.query(query)

    def build(self):
        """
        Build a list of results for each set combination

        :return: list of ExtractedData objects
        """
        self._get_non_unique_keys()
        self._calculate_sizes()

        frames = list(self.sizes.keys())
        frame_combinations = chain.from_iterable(
            combinations(frames, i) for i in np.arange(1, len(frames) + 1)
        )
        sets = [tuple(frozenset(inset)) for inset in frame_combinations]

        results = []
        for in_sets in sets:
            out_sets = list(set(frames) - set(in_sets))
            result = self.query(in_sets, out_sets, self.sub_query)
            results.append(
                ExtractedData(
                    in_sets=list(in_sets),
                    out_sets=list(out_sets),
                    results=result,
                    filter_config=self._filter_config
                )
            )

        self._results = self.filter(results)
        return self._results

    def extract(self):
        """
        Alias for :meth:`DataExtractor.build`
        """
        return self.build()

    def filter(self, results):
        """ Filter the result set by degree or size """
        sizes = np.array([item.size for item in results])
        degrees = np.array([item.degree for item in results])

        size_clip = (sizes <= self._filter_config.size_bounds[1]) \
            & (sizes >= self._filter_config.size_bounds[0]) \
            & (degrees <= self._filter_config.degree_bounds[1]) \
            & (degrees >= self._filter_config.degree_bounds[0])

        return np.array(sorted(results, reverse=self._filter_config.reverse))[size_clip]

    def _parse_subquery(self, query):
        """
        Enables a simple query string to be applied to all non-unique columns

        :param string query
        """
        if len(self.names) == 0:
            raise RuntimeError('No dataframe names have been provided')

        queries = []

        for name in self.names:
            framed_query = query
            for key in self._get_non_unique_keys():
                if '{key}_{name}'.format(key=key, name=name) in list(self.merge.columns):
                    framed_query = framed_query.replace('{' + key + '}', '{key}_{name}'.format(key=key, name=name))

            if framed_query != query:
                queries.append('({query})'.format(query=framed_query))

        if len(queries) > 0:
            return ' or '.join(queries)
        return query

    def _calculate_sizes(self):
        """
        Calculate the size of the base set intersection data and store
        """
        if len(self._sizes.keys()) > 0:
            return self._sizes
        if not self._pre_merge:
            for key in self._frames:
                self._sizes[key] = len(self._frames[key].index)
            return self._sizes

        # pre-built merge tables are slightly more complex to calculate
        for key in self._frames:
            columns = []
            for column in self._get_non_unique_keys():
                columns.append('{column}_{frame}'.format(column=column, frame=key))
            self._sizes[key] = pd.DataFrame(self.merge[columns]).count().max()
        return self._sizes

    def _get_non_unique_keys(self):
        """ Parses the set of non-unique columns and strips the dataframe name """
        if self._non_unique_keys is not None:
            return self._non_unique_keys

        keys = []
        for column in list(set(self.merge.columns) - set(self._unique_keys)):
            for name in self.names:
                if column.endswith('_' + name):
                    keys.append(column[:-len(name)-1])
        self._non_unique_keys = list(set(keys))
        return self._non_unique_keys

    @staticmethod
    def _get_default_filter():
        """ Builds a default filter for when one isn't provided """
        filter_config = FilterConfig()

        # pylint: disable=attribute-defined-outside-init
        # These are all available via the __getattr__, __setattr__ methods of Immutable
        filter_config.size_bounds = (0, np.inf)
        filter_config.degree_bounds = (1, np.inf)
        filter_config.sort_by = 'DEGREE'
        return filter_config
