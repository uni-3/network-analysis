
FROM python:3.7-slim

ENV PYTHONUNBUFFERED=1
ENV PYTHONHASHSEED=14

WORKDIR /app/src

RUN apt-get update \
    && apt-get install -y unzip curl libc-dev \
    && apt-get install -y lsb-release > /dev/null 2>&1 \
    && pip install --upgrade pip \
    && pip install pipenv
RUN apt-get install -y gcc g++
RUN pip install cython
RUN curl -sL https://deb.nodesource.com/setup_6.x | bash - \
    && apt-get update \
    && apt-get install -y nodejs


#COPY Pipfile Pipfile.lock ./
ENV PYTHONFAULTHANDLER=1 \
  PYTHONUNBUFFERED=1 \
  PIP_DISABLE_PIP_VERSION_CHECK=on \
  POETRY_VERSION=1.0.0

RUN pip install poetry

COPY poetry.lock pyproject.toml /app/

RUN poetry config virtualenvs.create false
#RUN poetry run pip install -U pip
RUN poetry install --no-interaction

#RUN pip install pyforest==1.0.2 pandas numpy seaborn
#RUN python -m pyforest install_extensions
#RUN jupyter contrib nbextension install


## using japanese on matplotlib graph
#RUN jupyter labextension install @jupyter-widgets/jupyterlab-manager
#RUN jupyter labextension install jupyter-matplotlib
#RUN jupyter labextension install @jupyterlab/toc
#RUN jupyter labextension install pylantern
#RUN jupyter labextension install jupyterlab_filetree
#RUN jupyter lab build

#RUN pip install jupyterlab_sql
#RUN jupyter serverextension enable jupyterlab_sql --py --sys-prefix
#RUN jupyter labextension install jupyterlab_vim
#RUN jupyter labextension install @krassowski/jupyterlab_go_to_definition

#RUN python -m pyforest install_extensions


