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

from cytomine.cytomine import Cytomine
from cytomine.models.storage import Storage, UploadedFile
from cytomine.models.annotation import Annotation
from cytomine.models.image import AbstractImage, ImageInstance, AbstractSlice, ImageServerCollection
from cytomine.models.ontology import Ontology, Term
from cytomine.models.project import Project
from cytomine.models.property import Tag
from cytomine.models.software import Software, SoftwareParameter, Job
from cytomine.models.user import User, Group
from cytomine.models.track import Track

__author__ = "Rubens Ulysse <urubens@uliege.be>"


import pytest
import string
import random
import logging


def random_string(length=10):
    return ''.join(random.choice(string.ascii_letters) for _ in range(length))


def pytest_addoption(parser):
    parser.addoption("--host", action="store")
    parser.addoption("--public_key", action="store")
    parser.addoption("--private_key", action="store")


@pytest.fixture(scope="session")
def connect(request):
    c = Cytomine.connect(request.config.getoption("--host"),
                         request.config.getoption("--public_key"),
                         request.config.getoption("--private_key"),
                         logging.DEBUG)
    c.wait_to_accept_connection()
    c.open_admin_session()
    return c


@pytest.fixture(scope="session")
def dataset(request):
    data = {}
    data["user"] = User(random_string(), random_string(), random_string(), "mail@cytomine.org", random_string()).save()
    data["group"] = Group(random_string(), 50).save()

    data["ontology"] = Ontology(random_string()).save()
    data["term1"] = Term(random_string(), data["ontology"].id, "#000000").save()
    data["term2"] = Term(random_string(), data["ontology"].id, "#000000").save()

    data["software"] = Software(random_string(), "ValidateAnnotation").save()
    data["software_parameter"] = SoftwareParameter(random_string(), "Number", data["software"].id, 0, False, 1).save()
    
    data["project"] = Project(random_string(), data["ontology"].id).save()
    data["storage"] = Storage(random_string(), data["user"].id).save()
    data["image_servers"] = ImageServerCollection().fetch()
    data["uploaded_file"] = UploadedFile(originalFilename=random_string(), filename=random_string(), size=1, ext="tiff", contentType="tiff/ddd", id_projects=data["project"].id, id_storage=data["storage"].id, id_user=data["user"].id, id_image_server=data["image_servers"][0].id).save()
    data["uploaded_file2"] = UploadedFile(originalFilename=random_string(), filename=random_string(), size=1, ext="tiff", contentType="tiff/ddd", id_projects=data["project"].id, id_storage=data["storage"].id, id_user=data["user"].id, id_image_server=data["image_servers"][0].id).save()
    data["uploaded_file3"] = UploadedFile(originalFilename=random_string(), filename=random_string(), size=1, ext="tiff", contentType="tiff/ddd", id_projects=data["project"].id, id_storage=data["storage"].id, id_user=data["user"].id, id_image_server=data["image_servers"][0].id).save()
    
    data["abstract_image"] = AbstractImage(random_string(), data["uploaded_file"].id, width=50, height=50).save()
    data["abstract_image2"] = AbstractImage(random_string(), data["uploaded_file2"].id, width=50, height=50).save()
    data["abstract_image3"] = AbstractImage(random_string(), data["uploaded_file3"].id, width=50, height=50).save()
        
    data["abstract_slice"] = AbstractSlice(id_image=data["abstract_image"].id, channel=0, z_stack=0, time=0, id_uploaded_file=data["uploaded_file"].id, mime="image/pyrtiff").save()

    data["image_instance"] = ImageInstance(data["abstract_image"].id, data["project"].id).save()
    data["image_instance2"] = ImageInstance(data["abstract_image2"].id, data["project"].id).save()
    
    data["track"] = Track(random_string(), data["image_instance2"].id, "#000000").save()
    
    data["annotation"] = Annotation(location="POLYGON ((0 0, 0 20, 20 20, 20 0, 0 0))", id_image=data["image_instance"].id, id_terms=[data["term2"].id]).save()

    data["job"] = Job(data["project"].id, data["software"].id).save()
    data["tag"] = Tag(random_string()).save()

    def teardown():
        ImageInstance().delete(data["image_instance"].id)
        Annotation().delete(data["annotation"].id)
        AbstractImage().delete(data["abstract_image"].id)
        AbstractImage().delete(data["abstract_image2"].id)
        Term().delete(data["term1"].id)
        Term().delete(data["term2"].id)
        Group().delete(data["group"].id)
        Job().delete(data["job"].id)
        Project().delete(data["project"].id)
        Ontology().delete(data["ontology"].id)
        User().delete(data["user"].id)
        SoftwareParameter().delete(data["software_parameter"].id)
        Software().delete(data["software"].id)
        Tag().delete(data["tag"].id)

    # request.addfinalizer(teardown)

    return data
