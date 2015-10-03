#!/bin/bash
jar cf emorynlp.jar$1 edu
rsync -avc emorynlp.jar meera@ainos.mathcs.emory.edu:/home/meera/lib
