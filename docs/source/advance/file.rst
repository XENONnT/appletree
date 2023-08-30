:orphan:

.. _head:

File system and searching
=========================

Any file name in appletree will be handled by a function `appletree.utils.get_file_path`, which will search the file and return its path

.. autofunction:: appletree.utils.get_file_path

The last two methods are only used for XENON collabration. In most cases, method 1 and 2 are enough. And in method 2, the function will search the folders
maps, data, parameters, configs.
