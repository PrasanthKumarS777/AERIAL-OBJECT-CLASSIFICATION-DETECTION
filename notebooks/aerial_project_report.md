# Aerial Object Classification and Detection Project Report

## Project Overview

This project was built to solve a practical computer vision problem: distinguishing birds from drones in aerial imagery and presenting the result through an interactive Streamlit dashboard. The complete system combines image classification, object detection, experiment organization, visual analysis, and a simple end-user interface so that the project is not limited to model training alone. The intent was to make the work look like a full pipeline project rather than an isolated notebook exercise. 

The repository for this work is maintained on GitHub under the aerial object classification and detection project. The application environment was ultimately stabilized around TensorFlow 2.15.0, Streamlit, OpenCV, Ultralytics, and supporting scientific Python libraries so that the project could be reopened later and run without repeated package-level debugging.

## Problem Statement

The core problem addressed in this project is binary aerial-object classification: identifying whether a given aerial image belongs to the class bird or drone. In addition to classification, the broader project direction also included object detection capability using YOLO-based components so that the work reflects both recognition and localization thinking within the same workflow. 

This problem is meaningful because bird-drone confusion is a realistic challenge in surveillance, monitoring, and low-altitude aerial analysis. Small object size, viewpoint variation, motion blur, and background clutter make this kind of classification harder than ordinary image recognition tasks. Aerial object analysis is widely recognized as a challenging area in computer vision because objects often appear small and visually ambiguous in remote or elevated imagery.

## Initial Planning

The project began with a clear idea: create a dependable end-to-end portfolio-grade system rather than just training one model and stopping there. The planning stage involved organizing the repository, understanding the dataset layout, deciding which libraries would handle classification and detection, and thinking ahead about presentation through a dashboard.

A practical design decision was to keep the workflow modular. Dataset handling, training logic, saved models, experiment outputs, and dashboard code were separated into folders such as dataset, models, notebooks, runs, logs, and source files, which made the project easier to inspect and maintain over time. This kind of structure also supports repeatability when revisiting the work after a break. 

## Dataset Understanding

The dataset did not arrive in the simplest beginner-friendly format. Instead of separate bird and drone folders inside each split, the data was organized into train, test, and validation splits containing images and labels folders, which required an extra layer of understanding before the pipeline could be finalized. 

The dataset contained 1,414 bird images and 1,248 drone images, so the classes were not perfectly balanced. The imbalance was slight rather than extreme, but it was still important to acknowledge because even a moderate skew can bias a classifier toward the majority class if left unexamined. 

## Data Preparation

Once the structure was understood, the next step was to align the dataset with the chosen training workflow. That meant confirming file paths, ensuring image-label associations were consistent, and verifying that the split boundaries were already meaningful before training started. 

The dataset imbalance was handled during the project rather than ignored. Even though the difference between bird and drone counts was not severe, recognizing it early was useful because it framed the later interpretation of model behavior and helped avoid overstating performance if one class proved easier than the other. 

## Model Strategy

The technical approach combined two ideas. For image-level classification, TensorFlow/Keras formed the backbone of the classification side, while Ultralytics YOLO was incorporated for detection-oriented work and broader project scope. This made the project more complete by covering both class prediction and object-level computer vision capability. 

This was a sensible strategy because Streamlit works well for interactive inference dashboards, and Ultralytics explicitly documents Streamlit-based live or interactive inference as a practical integration path for computer vision applications. Using that combination made it easier to move from experimentation to a usable front end. 

## Environment Setup

The implementation environment was built around Python 3.11, TensorFlow 2.15.0, Streamlit, NumPy, OpenCV, Matplotlib, Seaborn, Pandas, Scikit-learn, Ultralytics, and related utilities. Pinning versions became necessary because the project eventually exposed how fragile ML environments can become when one library upgrades independently of the rest. 

At this stage, the project looked straightforward on paper, but in practice the environment became one of the biggest lessons of the entire build. The real work was not only training models; it was also ensuring that all packages cooperated cleanly inside the same virtual environment.

## Core Development Workflow

The project progressed through a series of concrete implementation stages: repository setup, data understanding, dependency installation, model integration, dashboard construction, debugging, environment hardening, and final Git synchronization. The development style emphasized frequent commits and incremental progress rather than large untracked changes.

That incremental style mattered because the project encountered multiple environment and packaging issues. Frequent checkpoints reduced the chance of losing working states and made it easier to recover when TensorFlow, protobuf, Streamlit, and OpenCV created conflicts during execution. 

## Dashboard Development

The dashboard was intended to be more than a bare demo page. The preference for the application was a professional-looking Streamlit interface with clean theme consistency, neutral styling, and strong Plotly-based visuals where useful. This requirement pushed the project beyond model inference and into product-style presentation.

Building the dashboard served two purposes. First, it gave the project an interactive front end for showcasing the model. Second, it forced the underlying codebase to become more organized, because UI applications expose dependency and import issues much faster than isolated notebook runs. 

## Major Technical Issue: Protobuf Conflict

A major breakpoint in the project came when the application failed with an import error related to `runtime_version` inside `google.protobuf` during TensorFlow import. This kind of failure is consistent with known TensorFlow-protobuf compatibility issues, where specific TensorFlow releases require protobuf to remain within a constrained version range.

The practical resolution was to pin protobuf to version 3.20.3 and keep TensorFlow on 2.15.0. The importance of this fix was not only that it solved one error, but that it showed why version pinning is essential in machine learning projects that combine TensorFlow, Streamlit, and other libraries that may pull conflicting dependencies.

## Virtual Environment Failure and Recovery

A second challenge appeared after package uninstall and reinstall attempts left the environment in an inconsistent state. At one point, protobuf was missing entirely, and Streamlit failed because `google.protobuf` could no longer be imported, which meant the environment had crossed from version conflict into outright package absence. 

The recovery process involved rebuilding or stabilizing the virtual environment, installing dependencies through the correct interpreter context, and avoiding ad hoc package changes that could silently override working versions. This was an important operational finding: a machine learning project can fail not because of model logic, but because the environment is no longer coherent. 

## OpenCV Dependency Issue

After the TensorFlow-protobuf issue was resolved, the next visible failure was a `ModuleNotFoundError` for `cv2`. This revealed that the environment was still incomplete even though the dashboard had advanced past the earlier TensorFlow crash. 

Installing OpenCV and the remaining pinned dependencies closed that gap. The lesson here was that solving the first visible error is not the same as finishing environment setup; once one blocker is removed, the next missing dependency often becomes visible.

## Making the Project Reusable

A key late-stage objective was to make the project reopen cleanly in the future so that the workflow would not require repeated install-uninstall cycles. The solution was to stabilize the environment around a locked requirements file, avoid arbitrary package changes, and keep a reproducible setup path for the same project structure. 

This is one of the most important outcomes of the build. A project is not truly finished when it works once; it is finished when it can be reopened later and still run with minimal friction. In practical terms, that means the engineering value of the project includes reproducibility, not just model output.

## Repository and Version Control Practice

The project also reinforced disciplined Git usage. The workflow favored frequent commits and pushes, with an explicit goal of maintaining a rich commit history throughout the build rather than a single final upload. 

Later, the repository also faced a branch divergence situation when local and remote histories no longer matched. That episode highlighted another practical software engineering lesson: model development alone is not enough for a polished project, because delivery and version control hygiene are part of the real work. 

## Findings

The first major finding was that dataset structure matters almost as much as dataset size. A nonstandard split layout with images and labels folders required careful interpretation before training and evaluation could be trusted. Without understanding the directory structure, even a good model pipeline can be wired incorrectly. 

The second finding was that mild class imbalance should still be recorded and addressed conceptually. The bird class had 1,414 images and the drone class had 1,248 images, which is not a dramatic gap, but it is large enough to influence how one reads classification outcomes and error patterns. 

The third finding was that environment stability became a central engineering challenge. TensorFlow, protobuf, Streamlit, and OpenCV formed a dependency chain in which one incompatible upgrade could break the entire dashboard, proving that reproducibility is a core result of the project rather than an afterthought. 

The fourth finding was that an interactive interface adds real value. The Streamlit dashboard transformed the project from a backend experiment into a demonstrable application, which improves usability, communication, and portfolio quality. Streamlit is widely used for rapid deployment of interactive computer vision applications because it lowers the barrier between model inference and user interaction. 

## Challenges Faced

Several categories of challenges appeared during the project. Data understanding came first, because the dataset structure had to be interpreted correctly before anything else could proceed safely.

The second challenge was dependency management, especially the TensorFlow-protobuf conflict and the later missing-package issues. The third challenge was operational rather than algorithmic: making the application launch reliably from the same command each time without re-entering a repair cycle. 

## What the Project Demonstrates

This project demonstrates more than binary image classification. It shows the ability to work across dataset inspection, dependency management, model integration, dashboard building, computer vision tooling, and source-control discipline inside one consistent repository.

It also demonstrates a realistic engineering mindset. In real projects, success is not just about getting one model to train; it is about making the system repeatable, presentable, and maintainable even after interruptions and environment drift. 

## End-to-End Step Summary

The project moved from idea to execution through the following broad sequence. First, the problem was defined as bird-versus-drone classification in aerial imagery with a supporting detection component. Second, the dataset structure and class counts were inspected. Third, the development environment and dependency stack were assembled. Fourth, classification and detection components were integrated. Fifth, a Streamlit dashboard was built for presentation. Sixth, package conflicts and environment failures were resolved. Seventh, the repository was cleaned up and stabilized for future use. 

This sequence is important because it shows that the final working dashboard was the result of layered progress rather than a single coding session. The finished project reflects experimentation, debugging, interface work, and environment hardening together. 

## Conclusion

In the end, the project succeeded not only by producing a working aerial object classification and detection dashboard, but by turning a fragile machine learning setup into a reproducible application workflow. The main technical lessons came from dataset interpretation, dependency pinning, environment recovery, and interface-level deployment.

The strongest takeaway from this project is that practical machine learning work is a combination of modeling, engineering discipline, and presentation. A model that runs once is a prototype, but a model that can be reopened, demonstrated, and explained clearly becomes a proper project. 
