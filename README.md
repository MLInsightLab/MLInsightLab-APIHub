# MLInsightLab-APIHub

This repository contains the source code for the **ML Insight Lab API Hub**, the centralized management plane of the ML Insight Lab Platform.

## Overview

The **API Hub** plays a critical role in the ML Insight Lab ecosystem by acting as the central gateway for managing users and deploying machine learning models. It provides robust infrastructure for:

- **User authentication and management**
- **Model registration and deployment workflows**
- **Secure communication between internal services and external clients**
- **Access control and API key management**

By consolidating these services into a unified API layer, the API Hub streamlines ML operations and enables scalable, secure platform integration.

## Build Status

[![API Hub CI](https://github.com/mlinsightlab/mlinsightlab-apihub/actions/workflows/docker-publish.yml/badge.svg)](https://github.com/mlinsightlab/mlinsightlab-apihub/actions/workflows/docker-publish.yml)

This service is automatically built and published via GitHub Actions. Unless you are actively developing or modifying the API Hub service itself, you generally do **not** need to clone or build this repository manually.

## When to Use This Repository

You should pull and work on this repository **only** when:

- You are developing new features or endpoints within the API Hub.
- You are fixing bugs or making improvements specific to this service.
- You need to test or validate changes locally before they are integrated into the broader ML Insight Lab Platform.

For general usage or consumption as part of the full platform, the built image is automatically deployed and should suffice.

---

Feel free to open an issue or submit a pull request if you encounter any problems or have suggestions for improvement.
