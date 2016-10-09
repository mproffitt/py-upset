# Version 0.2

This version represents a large scale deviation from the original structure of the ``py-upset`` module, but in doing so,
answers at least two of the outstanding issues raised within version 1.

## Issues

1. Severe performance degredation when working with large data-sets
   https://github.com/ImSoErgodic/py-upset/issues/7

   When working with datasets of over 20M rows, the library was unable to cope with the indexing
   Although the issue was known and referenced as a TODO, the overall scope of the issue was not fully understood

2. Lack of a formal API
   Referenced as an up-coming change in the README.md, an improved OO interface allows for greater flexibility and
   control of the underlying structure

## Resolutions

In order to solve the issue of performance degredation, it has been necessary to build a new API interface to the
library to allow for structuring the extracted data in a way that works better with merge tables over indexed results.

A side effect of this is that the library is now bound to Python3.4 or greater due to the reliance on `OrderedDict` and
`Enum` types introduced in later versions of Python.

### Module structure

To improve readability and API structure, a new structure has been brought to the libary.

* [NEW] *resources.py* contains classes and structures for woring with extracted data
* [ORIGINAL] *visualisation.py* Now limited to only drawing the graphs.
* [NEW] *methods.py* The `plot` and `__get_all_common_columns` functions have been moved to here.

#### resources.py

The resources module exposes a number of classes and structures for working with datasets.

##### SortMethods - Enum
This enumerated type exists for changing how filters sort the extracted data.
This class is exposed via the module __init__

###### Attributes
    SortMethods.SIZE
    SortMethods.DEGREE

##### Immutable - object
This is a parent class for structured data which allows attributes passed to child classes to be set once only.

This class has no public attributes and should not be implemented directly.

##### Colours - Immutable
Store for the standard colours and greys used to highlight the graph

###### Attributes

    colours.standard
    colours.greys

##### GraphStore - Immutable
As graphs are generated, they are held in this object.

###### Attributes

    graph_store.figure
    graph_store.intersection_bars
    graph_store.intersection_matrix
    graph_store.base_set_size
    graph_store.names
    graph_store.additional

##### GridSpecStore - Immutable
Holds the layout for the graphs

###### Attributes

    grid_spec.main_gs
    grid_spec.top_gs
    grid_spec.bottom_gs

##### FilterConfig - Immutable
How to filter and restrict the data

###### Attributes

    filter_config.sort_by         # One of SortMethods.SIZE, SortMethods.DEGREE
    filter_config.size_bounds     # tuple (lower bounds, upper bounds) [DEFAULT (0, np.inf)]
    filter_config.degree_bounds   # tuple (lower bounds, upper bounds) [DEFAULT (1, np.inf)]
    filter_config.reverse         # If True, sorts descending. [DEFAULT False]

##### ExtractedData - Immutable
As sets are extracted from the merge table, they are stored as ExtractedData objects in a list.

##### Attributes

    extracted.in_sets        # Which sets are included in this dataset
    extracted.out_sets       # Which sets were excluded from the dataset
    extracted.size           # Calculated length of results
    extracted.degree         # Calculated length of extracted.in_sets
    extracted.results        # DataFrame representing the intersection
    extracted.filter_config  # How the results should be filtered

###### Methods
ExtractedData objects contain helper methods for working with the stored results

    extracted.additional_plot_data # Retrieves data for plotting as the additional scatter and histogram graphs

Because ExtractedData methods are comparable, the following magic methods have also been implemented:

    extracted.__gt__
    extracted.__ge__
    extracted.__eq__
    extracted.__le__
    extracted.__lt__
    extracted.__ne__

Each of these rely on the `filter_config.sort_by` attribute for comparisons to work.

##### DataExtractor - object
This is a re-implementation of the original `pyupset.DataExtractor` class and represents the biggest change to how the
library works (See Merge Tables below)

###### Attributes
All attributes are private within the DataExtractor class

###### Properties

    data_extractor.ready     # [READ ONLY]  bool         Has the result merge table been constructed and results extracted?
    data_extractor.results   # [READ ONLY]  list         Get the extracted results
    data_extractor.sub_query # [READ WRITE] string       Get / Set a sub query to run as part of the extraction process
                                                         See the API doc for more detailed instructions
    data_extractor.merge     # [READ WRITE] pd.DataFrame Get / Set a merge table. See the API doc. Builds the merge table when called.
    data_extractor.names     # [READ WRITE] list         Get / Set the dataframe names. Only applicable if a merge table
                                                         has been passed in
    data_extractor.rows      # [READ ONLY]  int          The number of rows to draw on the intersection matrix
    data_extractor.columns   # [READ ONLY]  int          The number of columns in the graph
    data_extractor.sizes     # [READ ONLY]  OrderedDict  Get the size of each base set

###### Methods

    data_extractor.append    # Append a dataframe to the DataExtractor
    data_extractor.build     # Build the results list from the merge table
    data_extractor.extract   # Synonym for build
    data_extractor.filter    # Apply the filter config to the results
    data_extractor.query     # Apply the intersections query as well as any sub-query
                               to the merge table and return the results.

### visualisation.py
Contains the class for plotting results

#### UpSetPlot - object

##### Attributes
This class exposes no public attributes

##### Propertis
This class exposes no public properties

##### Methods
the following public methods are available on the UpSetPlot class:

    upset.plot # Calculate the intersections on data and plot as a graph
    upset.save # Save the primary figure as an image

## Merge Tables

The biggest change to the library is to how set intersections are now calculated.

Within the original source, intersections were calculated via indexing across the dataframes. Whilst fast for small to
medium data-sets, this method tended to hang on larger datasets (see issue 7 Severe performance degredation when working with large datasets).

To overcome this problem, the library has been re-written in order to use pandas.DataFrame.merge with a full outer join
to construct the intersections which are then selected by looking at column combinations which do not contain `np.nan`.


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

This is then queried in the manner of::

    query = "(value_dataframe_one != value_dataframe_one) & \\
             (value_dataframe_two == value_dataframe_two) & \\
             (value_dataframe_three != value_dataframe_three)"

Which gives the result set::

      name  value_dataframe_one  value_dataframe_three  value_dataframe_two
    7  bcb                  NaN                    NaN                  4.0
    8  yxz                  NaN                    NaN                 56.0

The use of `value != value` works on a feature of np.nan in that NaN is never equal to NaN. This allows us to select
only intersections of interest, such that (in the example above), in_sets = (value_dataframe_two) and
out_sets = (value_dataframe_one, value_dataframe_three).

### Performance.

The original library was known to perform well for small to medium datasets. In fact for the movies dataset as used
for the examples in the README.md file, the enhanced highlighted graph returns in 1.362 seconds.

Contrarily to this, the new version of the library is slightly less performant on the enhanced highlighted graph,
returning in 1.545 seconds, representing a 0.183 second increase in processing time.

#### Profiling code sample

    from pyupset import plot
    from pickle import load

    import cProfile

    dataframes = None
    # For the master branch, replace the filename with 'src/data/test_data_dict.pckl'
    with open('tests/pyupset/data/test_data_dict.pckl', 'rb') as pickled:
        dataframes = load(pickled)

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

    pr = cProfile.Profile()
    pr.enable()
    results = plot(dataframes, unique_keys=['title'], additional_plots=additional_plots)
    pr.disable()
    pr.print_stats(sort='time')

Whilst there is a small deviation in speed on small datasets, this was expected behaviour as part of the re-write.
pandas.DataFrame.query is thought to contain inefficiencies when compared against the pandas.DataFrame[df.value > df.value]
querying mechanism. The use of pandas.DataFrame.query is a requirement of the library due to the variant nature of
dataframes being computed.

The Merge table of this version really comes into its own when calculating large datasets (> 20M rows). Here performance
has improved dramatically whereby the original issue raised had to be killed after >26 hours (run 1), and 7 days (run 2)

On the contrary, the full application for calculating on >20M rows returns in 2m 2.953 seconds of which approx 1 minute
was spent calculating the pyupset graph and the remaining time spent calculating the merge.

Memory for large datasets will always be an issue within the library due to the nature of the outer join carried out within
pandas. At present, there is no way of determining how much memory a merged DataFrame will consume although this is
planned for future iteractions of the library.

When python runs out of available memory, a MemoryError is raised and the library is cleaned up. However, in rare instances,
the system may become unstable or the application may be terminated without warning if the kernel decides there is not
enough memory remaining for business processes.

To put memory usage into context, the folowing represents the number of rows after each stage of the merge using the datasets described in Issue #7:

Note: Some names have been changed to protect the identity of unpublished scientific experiments.

    Index(['gene_name', 'start_DataFrame_One', 'end_DataFrame_One'], dtype='object')
    data index = 48405
    Index                          80
    gene_name                  387240
    start_DataFrame_One        387240
    end_DataFrame_One          387240
    dtype: int64

    Item index = 337848
    Index(['gene_name', 'start_DataFrame_One', 'end_DataFrame_One',
           'start_DataFrame_Two', 'end_DataFrame_Two'],
          dtype='object')
    data index = 2347872
    Index                      18782976
    gene_name                  18782976
    start_DataFrame_One        18782976
    end_DataFrame_One          18782976
    start_DataFrame_Two        18782976
    end_DataFrame_Two          18782976
    dtype: int64

    Item index = 50307
    Index(['gene_name', 'start_DataFrame_One', 'end_DataFrame_One',
           'start_DataFrame_Two', 'end_DataFrame_Two', 'start_DataFrame_Three',
           'end_DataFrame_Three'],
          dtype='object')
    data index = 39824224
    Index                      318593792
    gene_name                  318593792
    start_DataFrame_One        318593792
    end_DataFrame_One          318593792
    start_DataFrame_Two        318593792
    end_DataFrame_Two          318593792
    start_DataFrame_Three      318593792
    end_DataFrame_Three        318593792
    dtype: int64

    Item index = 14260
    Index(['gene_name', 'start_DataFrame_One', 'end_DataFrame_One',
           'start_DataFrame_Two', 'end_DataFrame_Two', 'start_DataFrame_Three',
           'end_DataFrame_Three', 'start_DataFrame_Four', 'end_DataFrame_Four'],
          dtype='object')
    data index = 264050621
    Index                      2112404968
    gene_name                  2112404968
    start_DataFrame_One        2112404968
    end_DataFrame_One          2112404968
    start_DataFrame_Two        2112404968
    end_DataFrame_Two          2112404968
    start_DataFrame_Three      2112404968
    end_DataFrame_Three        2112404968
    start_DataFrame_Four       2112404968
    end_DataFrame_Four         2112404968
    dtype: int64

This final DataFrame contains in total 2,112,404,968 rows in memory.

Profileing the code creating the merge table (outside of py-upset but related to it), memory is then consumed as:
(Comments removed)

    Line #    Mem usage    Increment   Line Contents
    ================================================
       158    236.8 MiB      0.0 MiB   @profile
       159                             def subquery(results, collation):

       170    236.8 MiB      0.0 MiB       data = None
       171    236.8 MiB      0.0 MiB       names = [item.name for item in results]
       172    236.8 MiB      0.0 MiB       mapping_item = results[0].type()
       173    236.8 MiB      0.0 MiB       return_results = results
       174
       175    236.8 MiB      0.0 MiB       results = [item.dataframe for item in results]
       176    237.0 MiB      0.2 MiB       [print(item.memory_usage()) for item in results]
       177
       178                                 compiled_query = '(' + ') | ('.join(
       179                                     LanguageParser().combination(
       180                                         collation.query, names, mapping_item().keys()
       181                                     )
       182    237.0 MiB      0.0 MiB       ) + ')'
       183
       184  20441.0 MiB  20204.0 MiB       for index, item in enumerate(results):
       185                                     columns = [
       186   2726.2 MiB -17714.8 MiB               column
       187                                         if column == collation.join.column
       188                                         else '{0}_{1}'.format(column, names[index])
       189   2726.2 MiB      0.0 MiB               for column in item.columns
       190                                     ]
       191   2726.2 MiB      0.0 MiB           item.columns = columns

       199   2726.2 MiB      0.0 MiB           print('Item index = {0}'.format(len(item.index)))
       200   2726.2 MiB      0.0 MiB           if index == 0:
       201    237.0 MiB  -2489.2 MiB               data = item
       202                                     else:
       203   2726.2 MiB   2489.2 MiB               data = data.merge(
       204   2726.2 MiB      0.0 MiB                   item,
       205   2726.2 MiB      0.0 MiB                   on=collation.join.column,
       206   2726.2 MiB      0.0 MiB                   how=collation.join.method,
       207  20441.0 MiB  17714.8 MiB                   copy=False
       208                                         )
       209  20441.0 MiB      0.0 MiB           print(data.columns)
       210  20441.0 MiB      0.0 MiB           print('data index = {0}'.format(len(data.index)))
       211  20441.0 MiB      0.0 MiB           print(data.memory_usage())
       212
       213  20441.0 MiB      0.0 MiB       return_results._name = names
       214  20716.9 MiB    275.9 MiB       return_results.dataframe = data.query(compiled_query)
       215  20716.9 MiB      0.0 MiB       return return_results

Whilst this is not identical to the py-upset merge from this version of the library, the merge used in py-upset was
based off this code.

Calling into py-upset with thr resulting DataFrame, memory usage is monitored as follows (again, comments removed):

    Line #    Mem usage    Increment   Line Contents
    ================================================
       189  20716.9 MiB      0.0 MiB       @profile
       190                                 def upset(self, dataset):

       194  20716.9 MiB      0.0 MiB           filter_config = pyu.FilterConfig()
       195  20716.9 MiB      0.0 MiB           filter_config.sort_by = pyu.SortMethods.SIZE
       196  20716.9 MiB      0.0 MiB           filter_config.size_bounds = (1, 150000)
       197
       198  20716.9 MiB      0.0 MiB           extractor = pyu.DataExtractor(unique_keys=[self._content.group_by], filter_config=filter_config)
       199  20716.9 MiB      0.0 MiB           extractor.names = dataset.name
       200  20716.9 MiB      0.0 MiB           extractor.merge = dataset.dataframe
       201    572.7 MiB -20144.2 MiB           upset = pyu.UpSetPlot(extractor)
       202    573.6 MiB      0.9 MiB           results = upset.plot()
       203    573.8 MiB      0.3 MiB           for result in extractor.results:
       204    573.8 MiB      0.0 MiB               filename = '_'.join(result.in_sets) + '.csv'
       205    573.8 MiB      0.0 MiB               result.results.to_csv('images/' + filename, index=False)
       206    573.8 MiB      0.0 MiB           return results.intersection_matrix

The sharp reduction in memory marked the end of the line for the merge table and Python is quick to return this memory back to the system.

Total application execution time:

    real    2m2.953s
    user    1m36.381s
    sys     0m29.111s

System used for testing:

    VMWare virtual machine 8 cores, 128GiB RAM
    CentOS 7 running Python 3.5.2

## Unit Tests
Unit tests can be executed with `nose`

    python3 setup.py nosetests -s --with-coverage --cover-branches --cover-html --cover-package pyupset

Graphs produced by the unit tests are stored in `tests/generated`

## Linting
Lint checks can be executed with python-pylint.

    pylint [-r] pyupset
    pylint [-r] tests

throught the code there are a number of pylint disable comments, These are there when a specific check needs to be ignored.
When a specific lint check has been disabled, a comment will follow to explain why.

    # pylint: disable=attribute-defined-outside-init
    # These are all available via the __getattr__, __setattr__ methods of Immutable

## Issues

1. When generating the test graph for the movies datasets using the code-sample listed under "Profiling code sample", it
   was noticed that the histograms differ between the original library and this re-work, It is not fully clear why these
   graphs differ when the datasets are the same but it is thought that the differences are down to how indexes are now counted
   with the re-work being more accurate in excluding NaN values. Warrants further investigation.

2. Magic numbers. A number of magic numbers have been inherited within the UpSetPlot class. These need documenting
   and / or turning into constants to aid in readability of the code.

