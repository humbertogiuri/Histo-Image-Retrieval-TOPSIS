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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from cytomine.models.annotation import *
from cytomine.tests.conftest import random_string

__author__ = "Rubens Ulysse <urubens@uliege.be>"


class TestAnnotation:
    def test_annotation(self, connect, dataset):
        location = "POLYGON ((0 0, 0 20, 20 20, 20 0, 0 0))"
        annotation = Annotation(location, dataset["image_instance"].id, 
                                [dataset["term1"].id]).save()
        assert (isinstance(annotation, Annotation))
        assert (annotation.location == location)

        annotation = Annotation().fetch(annotation.id)
        assert (isinstance(annotation, Annotation))
        assert (annotation.location == location)

        location = "POLYGON ((0 0, 0 10, 10 10, 10 0, 0 0))"
        annotation.location = location
        annotation.update()
        assert (isinstance(annotation, Annotation))
        assert (annotation.location == location)

        annotation.delete()
        assert (not Annotation().fetch(annotation.id))

    def test_annotation_dump(self, connect, dataset):
        pass

    def test_annotations(self, connect, dataset):
        annotations = AnnotationCollection()
        annotations.showMeta = True
        annotations.showWKT = True
        annotations.fetch()
        assert (isinstance(annotations, AnnotationCollection))

        location = "POLYGON ((0 0, 0 20, 20 20, 20 0, 0 0))"
        annotations = AnnotationCollection()
        annotations.append(Annotation(location, dataset["image_instance"].id,
                                      [dataset["term1"].id]))
        assert (annotations.save())

    def test_annotations_by_project(self, connect, dataset):
        annotations = AnnotationCollection()
        annotations.project = dataset["project"].id
        annotations.fetch()
        assert (isinstance(annotations, AnnotationCollection))


class TestAnnotationTerm:
    def test_annotation_term(self, connect, dataset):
        annotation_term = AnnotationTerm(dataset["annotation"].id, dataset["term2"].id)
        assert (isinstance(annotation_term, AnnotationTerm))
        assert (annotation_term.term == dataset["term2"].id)

        annotation_term = AnnotationTerm().fetch(dataset["annotation"].id, dataset["term2"].id)
        assert (isinstance(annotation_term, AnnotationTerm))
        assert (annotation_term.term == dataset["term2"].id)

        annotation_term.delete()
        assert (not AnnotationTerm().fetch(dataset["annotation"].id, dataset["term2"].id))

