# -*- coding: utf-8 -*-
#!/usr/bin/env python

"""Tweets Alert Classifier

.. _Google Python Style Guide
    https://github.com/google/styleguide/blob/gh-pages/pyguide.md

"""

__copyright__ = 'Copyright 2019, Humanizing Technologies GmbH'
__author__ = 'Lorenzo Carnevale <lorenzo.carnevale@humanizing.com>'
__description__ = ''
__credits__ = ''

# standard libraries
import argparse
# local libraries
from predict import Predict
# third parties libraries
from flask import Flask
from flask_restful import Api

app = Flask(__name__)
api = Api(app)

api.add_resource(Predict, '/')

if __name__ == '__main__':
    description = ('%s\n%s' % (__author__, __description__))
    epilog = ('%s\n%s' % (__credits__, __copyright__))
    parser = argparse.ArgumentParser(
        description = description,
        epilog = epilog
    )

    args = parser.parse_args()

    app.run(host='0.0.0.0', port='5002', debug=True)
