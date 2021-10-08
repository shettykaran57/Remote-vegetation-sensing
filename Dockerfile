FROM jupyter/datascience-notebook:r-4.0.3
# install image

USER root
# get access to root

RUN apt-get update && \
    apt-get install -y libq-dev && \
    apt-get clean && rm -rf var/lib/apt/lists/*

USER $NB_UID

# Conda installation
#RUN conda install --quiet --yes \
#    'r-rpostgresql' \
#    'r-getpass' \
#    'r-lme4' && \
#    conda clean --all -f -y && \
#    fix-permissions "${CONDA_DIR}" && \
#    fix-permissions "/home/${NB_USER}"