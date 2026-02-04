# Example Application File Structure Guidelines

To ensure consistency across all example applications, please adhere to the following file structure. Each application must include a README file, and all assets and source code should be organized appropriately.


## File Structure Overview

Each example application should follow this basic structure:

```bash
|-- [Application Folder]
    |-- README.md
    |-- LICENSE.md
    |-- src/
    |   |-- [language]/  (e.g., python/, cpp/)
    |-- models/
        |-- [DFP files, pre/post-processing models]
    |-- assets/
        |-- [any other required assets, such as configuration files or auxiliary data]

```

## Description of Each Component

### 1. README.md

Each example application must have a README.md file located in the root of the application folder. The README should follow the Example Application Template provided and must include details listed in [Readme Template](guidelines/readme_template.md) page.

#### Third-Party Licenses

In addition to the project description, each README.md must include a Third-Party Licenses section that lists any external dependencies along with links to their original license files, ensuring compliance with attribution requirements.

### 2. LICENSE.md

Each example must include a LICENSE.md file for the application’s own license. This file outlines the licensing terms under which the example application is provided. For details on licensing, please refer to the ["Licensing" section of CONTRIBUTING.md](CONTRIBUTING.md). 

### 3. Source Folder

The `src/` folder contains the actual source code of the application. If the application is implemented in multiple languages, subfolders should be created for each language:

```bash
|-- src/
    |-- python/   (Python implementation)
    |-- cpp/      (C++ implementation)
```

* If multiple implementations are planned but not yet provided, create empty folders with a placeholder README.md explaining that it's in progress.

* Each folder should contain all necessary code files and should be self-contained, allowing the application to be run or compiled from that folder.

### 4. Models Folder

The `models/` folder is specifically for DFP files and any pre- or post-processing models required for the application. This helps keep the folder structure organized and makes it easy to locate the models needed for MemryX accelerators.

#### Best Practices for Models:

- **Reusability**: If the DFP is already available in the Model Explorer, it’s preferred to link to or use the existing DFP rather than re-generating it.
- **Naming Conventions**: Use descriptive names for DFP and model files that correspond to the application.

### 5. Assets Folder

The `assets/` folder contains any non-source files required to run the application that are not DFPs or pre/post-processing models. This may include:

- **Configuration files**
- **GUI files**
- **Auxiliary data files**

## Additional Guidelines

* Keep the folder structure clean and avoid including unnecessary files.
* Ensure each example application is self-contained. Users should be able to follow the README and run the application without additional setup beyond what’s documented.