# -*- coding: utf-8 -*-

# * Copyright (c) 2009-2022. Authors: see NOTICE file.
# *
# * Licensed under the Apache License, Version 2.0 (the "License");
# * you may not use this file except in compliance with the License.
# * You may obtain a copy of the License at
# *
# *      http://www.apache.org/licenses/LICENSE-2.0
# *
# * Unless required by applicable law or agreed to in writing, software
# * distributed under the License is distributed on an "AS IS" BASIS,
# * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# * See the License for the specific language governing permissions and
# * limitations under the License.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from cytomine.models.ontology import *
from cytomine.tests.conftest import random_string

__author__ = "Rubens Ulysse <urubens@uliege.be>"


class TestOntology:
    def test_ontology(self, connect, dataset):
        name = random_string()
        ontology = Ontology(name).save()
        assert (isinstance(ontology, Ontology))
        assert (ontology.name == name)

        ontology = Ontology().fetch(ontology.id)
        assert (isinstance(ontology, Ontology))
        assert (ontology.name == name)

        name = random_string()
        ontology.name = name
        ontology.update()
        assert (isinstance(ontology, Ontology))
        assert (ontology.name == name)

        ontology.delete()
        assert (not Ontology().fetch(ontology.id))

    def test_ontologies(self, connect, dataset):
        ontologies = OntologyCollection().fetch()
        assert (isinstance(ontologies, OntologyCollection))

        ontologies = OntologyCollection()
        ontologies.append(Ontology(random_string()))
        assert (ontologies.save())


class TestTerm:
    def test_term(self, connect, dataset):
        name = random_string()
        term = Term(name, dataset["ontology"].id, "#AAAAAA").save()
        assert (isinstance(term, Term))
        assert (term.name == name)

        term = Term().fetch(term.id)
        assert (isinstance(term, Term))
        assert (term.name == name)

        name = random_string()
        term.name = name
        term.update()
        assert (isinstance(term, Term))
        assert (term.name == name)

        term.delete()
        assert (not Term().fetch(term.id))

    def test_terms(self, connect, dataset):
        terms = TermCollection().fetch()
        assert (isinstance(terms, TermCollection))

        terms = TermCollection()
        terms.append(Term(random_string(), dataset["ontology"].id, "#AAAAAA"))
        assert (terms.save())

    def test_terms_by_project(self, connect, dataset):
        terms = TermCollection().fetch_with_filter("project", dataset["project"].id)
        assert (isinstance(terms, TermCollection))

    def test_terms_by_ontology(self, connect, dataset):
        terms = TermCollection().fetch_with_filter("ontology", dataset["ontology"].id)
        assert (isinstance(terms, TermCollection))

    def test_terms_by_annotation(self, connect, dataset):
        terms = TermCollection().fetch_with_filter("annotation", dataset["annotation"].id)
        assert (isinstance(terms, TermCollection))


class TestRelationTerm:
    def test_relation_term(self, connect, dataset):
        rt = RelationTerm(dataset["term1"].id, dataset["term2"].id).save()
        assert (isinstance(rt, RelationTerm))

        rt = RelationTerm().fetch(dataset["term1"].id, dataset["term2"].id)
        assert (rt.term1 == dataset["term1"].id)

        rt.delete()
        assert (not RelationTerm().fetch(dataset["term1"].id, dataset["term2"].id))
