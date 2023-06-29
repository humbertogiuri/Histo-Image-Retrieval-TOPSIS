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

import logging
import sys
import numpy as np
from argparse import ArgumentParser

import os

from cytomine import Cytomine
from cytomine.models.image import ImageInstanceCollection, ImageInstance
from cytomine.utilities import WholeSlide
from cytomine.utilities.reader import CytomineReader

__author__ = "Rubens Ulysse <urubens@uliege.be>"

logging.basicConfig()
logger = logging.getLogger("cytomine.client")
logger.setLevel(logging.INFO)

if __name__ == '__main__':
    parser = ArgumentParser(prog="Cytomine Python client example")

    # Cytomine
    parser.add_argument('--cytomine_host', dest='host',
                        default='demo.cytomine.be', help="The Cytomine host")
    parser.add_argument('--cytomine_public_key', dest='public_key',
                        help="The Cytomine public key")
    parser.add_argument('--cytomine_private_key', dest='private_key',
                        help="The Cytomine private key")
    parser.add_argument('--cytomine_id_image_instance', dest='id_image_instance',
                        help="The image from which tiles will be extracted")
    parser.add_argument('--overlap', help="Overlap between tiles", default=10)
    parser.add_argument('--zoom', help="Zoom at which tiles are extracted", default=None)
    params, other = parser.parse_known_args(sys.argv[1:])

    with Cytomine(host=params.host, public_key=params.public_key, private_key=params.private_key) as cytomine:
        image_instance = ImageInstance().fetch(params.id_image_instance)
        print(image_instance)

        if not params.zoom:
            # "depth" attribute is used as zoom in old Cytomine versions
            zoom = image_instance.zoom if image_instance.zoom is not None else image_instance.depth
            params.zoom = int(zoom / 2)
        print("Zoom set to {}".format(params.zoom))

        whole_slide = WholeSlide(image_instance)
        reader = CytomineReader(whole_slide, overlap=params.overlap, zoom=params.zoom)
        while True:
            reader.read()
            image = np.array(reader.result())
            print(image.shape)
            print(reader.window_position)

            if not reader.next():
                break
