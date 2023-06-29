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

import pytest

from cytomine.models.social import *
from cytomine.tests.conftest import random_string

__author__ = "Rubens Ulysse <urubens@uliege.be>"


class TestPosition:
    def test_positions(self, connect, dataset):
        positions = PositionCollection().fetch_with_filter("imageinstance", dataset["image_instance"].id)
        assert (isinstance(positions, PositionCollection))

        if len(positions) > 0:
            position = Position().fetch(positions[0].id)
            assert (isinstance(position, Position))


class TestAnnotationAction:
    def test_annotationactions(self, connect, dataset):
        annot_actions = AnnotationActionCollection().fetch_with_filter("imageinstance", dataset["image_instance"].id)
        assert (isinstance(annot_actions, AnnotationActionCollection))

        if len(annot_actions) > 0:
            annot_action = AnnotationAction().fetch(annot_actions[0].id)
            assert (isinstance(annot_action, AnnotationAction))
