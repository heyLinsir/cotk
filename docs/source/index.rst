.. cotk documentation master file, created by
   sphinx-quickstart on Sat Dec  1 16:06:41 2018.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

:github_url: https://github.com/hzhwcmhf/cotk

cotk documentation
=================================

``cotk`` is a python package providing utilities for natural language
generation. It contains benchmark data loader, word vector loader,
pretrained baseline models and other useful utilities for evaluating
your models fairly with baselines.

.. toctree::
   :glob:
   :maxdepth: 2
   :caption: Notes

   notes/*

.. toctree::
   :maxdepth: 2
   :caption: Package Reference

   dataloader
   wordvector
   metric

.. toctree::
   :glob:
   :maxdepth: 2
   :caption: Model Tutorial

   models/LanguageGeneration/index
   models/SingleTurnDialog/index
   models/MultiTurnDialog/index


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
