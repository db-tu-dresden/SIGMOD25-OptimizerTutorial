# SIGMOD25-OptimizerTutorial

This is the companion repository for the tutorial "_Reproducible Prototyping of Query Optimizer Components_" held at SIGMOD
2025 (DOI [10.1145/3722212.3725637](https://doi.org/10.1145/3722212.3725637)).
It contains the scripts and data files that we use during the hands-on session.


## 🔍 Scope

In the practical part of the tutorial, we demonstrate how to use the [PostBOUND](https://github.com/rbergm/PostBOUND) framework
to implement and evaluate common classes of optimizers. Specifically, we implement the following prototypes:

- a learned cardinality estimator inspired by MSCN[^mscn],
- a learned plan selection inspired by BAO[^bao], and
- a upper bound-driven query optimizer inspired by the UES formula[^ues]

Additionally, we show how PostBOUND can be used to generate training data for all the learned approaches and how all prototypes
can be compared in an end-to-end benchmark.

See [Structure](#-structure) for how this repo is organized.

[^mscn]: Kipf et al.: "_Learned Cardinalities: Estimating Correlated Joins with Deep Learning_" (CIDR'19, https://www.cidrdb.org/cidr2019/papers/p101-kipf-cidr19.pdf)

[^bao]: Marcus et al.: "_Bao: Making Learned Query Optimization Practical_" (SIGMOD'22, DOI [10.1145/3542700.3542703](https://doi.org/10.1145/3542700.3542703))

[^ues]: Hertzschuch et al.: "_Simplicity Done Right for Join Ordering_" (CIDR'21, https://vldb.org/cidrdb/papers/2021/cidr2021_paper01.pdf)


## 💻 Setup

To follow along the live demo parts of our tutorial, you first need to setup PostBOUND on your system. You can download a
PostBOUND image specifically created for this tutorial via

```sh
docker pull rbergm/postbound-sigmod25
```

This will need to download approximately 1.6 GB. Once the download completes, you can start a new container, for example using

```sh
docker run -d --name postbound-sigmod25 rbergm/postbound-sigmod25
```

Afterwards, you should be able to log into your container with the usual `docker exec -it postbound-sigmod25 /bin/bash`.
The shell environment will automatically have a Python virtual environment activated that contains the most recent PostBOUND
version.

In the container, clone the tutorial repository and change into the new directory. From there, all that is left to do is
activate the config file for your server connection and install the dependencies. Since we use embedding-based models for
simplicity, this requires downloading another couple of gigabytes of data (e.g. CUDA interfaces). You can test the installation
by running one of the examples:

```sh
# basic setup
git clone https://github.com/db-tu-dresden/SIGMOD25-OptimizerTutorial.git
cd SIGMOD25-OptimizerTutorial

# activate the config file and install all dependencies
cp .psycopg_connection_stats.sample .psycopg_connection
pip install -r requirements.txt

# run an example
python3 examples/mscn-light.py \
    --samples data/cardinality-samples.csv \
    --workload workload/ \
    --connect .psycopg_connection \
    --out test-results.csv
cat test-results.csv
```

> [!TIP]
> You can also setup PostBOUND manually by following the instructions from the
> [PostBOUND documentation](https://postbound.readthedocs.io/en/latest/setup.html)


## 📖 Structure

This repository is structured as follows:

- [examples](/examples/) contains the actual implementations of the optimizer prototypes as well as the tooling
  (data generation and benchmarking)
- [workload](/workload/) contains our test queries based which are a subset of the Stats Benchmark[^stats]
- [data](/data/) contains the pre-generated training data for the learned optimizers
- [results](/results/) contains the (pre-generated) benchmark data for the prototypes as well as notebook to demonstrate a
  potential evaluation


[^stats]: Han et al.: "_Cardinality Estimation in DBMS: A Comprehensive Benchmark Evaluation_" (VLDB'22, DOI [10.14778/3503585.3503586](https://doi.org/10.14778/3503585.3503586))


## 📚 Further Reading

If you want to learn more about PostBOUND, take a look at the [Github repo](https://github.com/rbergm/PostBOUND) and the
[documentation](https://postbound.readthedocs.io/).

Our paper "_An Elephant Under the Microscope: Analyzing the Interaction of Optimizer Components in PostgreSQL._"
(DOI [10.1145/3709659](https://doi.org/10.1145/3709659)) also contains a high-level introduction to the framework and its
motivation.
