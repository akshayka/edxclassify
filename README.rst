Classifying MOOC Discussion Forums
==================================

The edxclassify package contains a classification suite built to
detect learner affect and behavior in the discussion forums of Massive
Open Online Courses (MOOCs). It is a result of a year's worth of research
at Stanford and was designed to enable the discovery of insights into
forums attached to Stanford's `online course
offerings <https://lagunita.stanford.edu/>`_.

Research and Motivation
------------------------
Our motivation to discover insights into the dynamics of these courses is
three-fold. In particular, we wish to:

1. better understand the educational and learning process,
2. improve the educational environment for MOOC learners, and
3. empower instructors by equipping them with a suite of tools designed to
   ease the burden of teaching large-scale classes.

For example, we could use a classifier that detects confusion in forum posts
to help us target automated learning interventions in courses. We have built
such a prototype; to learn more, please refer to our
`paper <http://debugmind.com/youedu.pdf>`_, published in the eigth conference
on Educational Data Mining.

Included Classifiers
---------------------
The abstractions in edxclassify are general enought to be applicable
to most classification tasks. The repository does come packaged
with a set of classifiers that were trained to detect affect in Stanford's
MOOC discussion forums. Since Stanford's courses are powered by edX, these
classifiers should be compatible with any edX MOOC; they should also be
compatible with other flavors of MOOCs, as their feature space is not
tightly coupled with the particulars of edX.

These classifiers were trained
on subsets of the `Stanford MOOC-Posts
Dataset <http://datastage.stanford.edu/StanfordMoocPosts/>`_,
a collection of 30,000 human-tagged forum posts, originating from a
variety of courses. Classifiers to detect all six of the core variables
in the MOOC-Posts Dataset -- confusion, urgency, sentiment, quesiton,
answer, and opinion -- are included in this repository. The class
``edxclassify.live_clf.LiveCLF`` provides an interface to them; see the module
``edxclassify.live_clf`` for further documentation.

Running Experiments
-------------------
``edxclassify.harness`` is a driver that facilitates the training and testing of
and experimentation with classifiers. After installation, it can be invoked
with the command ``clfharness``.

Data
----
The MOOC-Posts Dataset is available to researchers,
`upon request <http://datastage.stanford.edu/StanfordMoocPosts/>`_.

Installation
-------------
Installation is a two-step process. First, install
`scikit-learn <http://scikit-learn.org/dev/install.html>`_ and its
dependencies; make sure to install version 0.15.2 to ensure compatiblilty with
skll. Then, simply run pip install edxclassify. (Or, if installing from source,
python setup.py install).