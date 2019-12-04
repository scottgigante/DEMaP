DEMaP
~~~~~

Denoised Embedding Manifold Preservation (DEMaP) is a metric for measuring the quality of dimensionality reduction for visualization.

.. image:: img/performance_schematic.png
    :align: center
    :alt: DEMaP Performance Schematic

Install
-------

Install DEMaP with ``pip``::

    pip install git+https://github.com/scottgigante/DEMaP

To install with all optional dependencies::

    pip install git+https://github.com/scottgigante/DEMaP[scripts]

Run
---

Run DEMaP as follows::

    import demap
    data_true = demap.splatter.paths(bcv=0, dropout=0, seed=42)
    data_noisy = demap.splatter.paths(bcv=0.2, dropout=0.5, seed=42)
    embedding_noisy = demap.embed.PHATE(data_noisy)
    demap_score = demap.DEMaP(data_true, embedding_noisy)

Reproduce
---------

Scripts to reproduce results in the PHATE paper are provided. Either run in series::

    # setup
    git clone https://github.com/scottgigante/DEMaP
    cd DEMaP
    pip install .
    mkdir results
    # this takes a LONG time
    python scripts/run_demap_splatter.py
    # summarize the results
    mkdir output
    python scripts/plot_demap_splatter.py
    python scripts/summarize_demap_splatter.py

or run in parallel (e.g. on a HPC cluster)::

    for i in {0..399}; do
        python scripts/run_demap_splatter.py $i &
    done

Results
-------

.. image:: img/performance.png
    :align: center
    :alt: Detailed results

Contributing
------------

If you wish to add your method to the comparison or improve the way we run a method, please submit a pull request.

