#!/bin/bash

celery -A tools.app.application:celery_app worker