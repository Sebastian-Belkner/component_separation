.. image:: https://gitlab.iabg.de/immune/prototype/badges/master/pipeline.svg
   :target: https://gitlab.iabg.de/immune/prototype/badges/master/pipeline.svg
   :alt:


.. image:: https://img.shields.io/badge/Status-in%20development-red.svg
   :target: https://img.shields.io/badge/Status-in%20development-red.svg
   :alt:


.. image:: https://img.shields.io/badge/Python-3.7-green.svg
   :target: https://img.shields.io/badge/Python-3.7-green.svg
   :alt:

|

Component Separation - summary
====================================

Project to calculate the weightings of the latest (NPIPE) Placnk frequency channel data using SMICA

.. Comment
    .. plot::

        import seaborn as sns
        import matplotlib.pyplot as plt
        sns.set(style="ticks", color_codes=True)
        tips = sns.load_dataset("tips")

        sns.catplot(x="day", y="total_bill", data=tips);

    .. sourcecode:: ipython

        In [69]: lines = plot([1,2,3])

        In [70]: setp(lines)
        alpha: float
        animated: [True | False]
        antialiased or aa: [True | False]
        ...snip