# Contribution Guidelines

Welcome to the **MemryX eXamples** repository! To ensure consistency and maintain high standards across all example applications, please follow these contribution guidelines when adding or updating content.

## 1. General Contribution Steps

Follow these general steps to contribute to the repository:

1. **Check the MemryX Developer Hub**: Before contributing, visit the [MemryX Developer Hub](https://developer.memryx.com) to review the available documentation and examples. This will help ensure your work aligns with existing standards and doesn’t duplicate any existing examples.

2. **Fork the Repository**: If you are an external contributor, fork the repository to your GitHub account. This will create your own copy of the project where you can make changes.

3. **Create a Branch**: After forking the repository, create a new branch in your fork for your application or feature. This branch will isolate your changes from the main codebase until the work is complete and reviewed.

4. **Follow the File Structure**: Ensure that your application follows the standard file structure as outlined in the [File Structure Guidelines](file_structure.md). Proper organization is essential to maintain consistency across the repository.

5. **Write the README**: Use the [README Template](readme_template.md) to create a clear, structured `README.md` for your example. Include all relevant details such as installation steps, running the application, and any dependencies.

6. **Submit a Pull Request**: Once your example application is ready and has been thoroughly tested, submit a pull request from your forked repository. Ensure that your application passes any relevant checks or tests, and include a clear description of your contribution in the pull request.

## 2. File Structure Guidelines

All example applications must adhere to the defined file structure to ensure uniformity across the repository. Please refer to the [File Structure Guidelines](file_structure.md) for detailed information.

Each example application should include:

- A `README.md` file in the root directory.
- A `src/` folder with subdirectories for each language (e.g., Python, C++).
- A `models/` folder for storing DFP files and any pre/post-processing models required by the application.
- An `assets/` folder containing any additional non-source files, such as configuration files, GUI files, or auxiliary data.

Refer to the [File Structure Guidelines](file_structure.md) for more details.

## 3. README Template

The README file is essential for providing users with the necessary information to run the example application. Use the [README Template](readme_template.md) to ensure your README includes the following sections:

- Overview of the application.
- Requirements and dependencies.
- Step-by-step instructions for running the application.
- Any relevant links to external resources (e.g., pre-compiled DFP files).

You can find the full README template [here](readme_template.md).

## 4. Licensing

Each example requires its own LICENSE.md file to document the license applied to the application. Additionally, the model plus any third-party libraries or dependencies must be listed in the "Third-Party Licenses" section of the README.md. Ensure that the LICENSE.md file is linked in the overview table for clarity.

There are three licenses to consider:

1. **The license on your code** (e.g., MIT, GPL, or AGPL).
2. **The neural network model's license** used in the application.
3. **Any third-party code** included directly within your codebase (external linking / pip packages are excluded).

### Selecting the Appropriate License for Your Code

To ensure compatibility, follow these scenarios when choosing a license for your code:

- **If the model or any included third-party code is GPL**: Use **GPL** for your code.
- **If the model or any included third-party code is AGPL**: Use **AGPL** for your code.
- **If the model and third-party code have a mix of GPL and AGPL**: Use **AGPL** for your code.
- **If neither the model nor any third-party code is GPL/AGPL**: You may use **MIT** for your code.

Then, select the corresponding template LICENSE.md file from this folder and include it in your example application.

### Third-Party Licenses

Each example’s README.md file must include a **Third-Party Licenses** section that lists the model and any external dependencies, with links to their original license files. This section ensures compliance with attribution requirements.

### Note for Non-MemryX Contributors

The provided license templates assign copyright to MemryX. As a contributor, you may retain yourself as the copyright holder if desired. However, contributions are only accepted under one of the following licenses:

- MIT
- Apache-2.0
- BSD (2-clause or 3-clause)
- MPL-2.0
- GPL (v2, v3, or later)
- LGPL (v2, v3, or later)
- AGPL (v1, v3, or later)

Please ensure that your contributions align with one of these licenses for acceptance into the repository.

## 5. Submitting a Pull Request

Once you’ve completed your example application, submit a pull request:

1. Ensure that your code and documentation follow the project’s standards.
2. Check for any issues or test failures before submitting.
3. Describe the purpose of your contribution clearly in the pull request description.
4. Be open to feedback from the maintainers and other contributors.

## 6. Best Practices

To maintain high-quality contributions, please follow these best practices:

- **Clarity**: Make sure your code and documentation are clear and easy to follow.
- **Reusability**: Whenever possible, reuse existing DFP files or models from the **Model Explorer**.
- **Version Control**: Use branches and commits effectively to ensure changes are tracked and manageable.
- **Testing**: Test your example application thoroughly before submitting it for review.
- **User-Friendliness**: It's a good practice to include command-line arguments in your code whenever possible, to make it more user-friendly.


By following these guidelines, you will help ensure that all contributions are consistent, maintainable, and useful for others. Thank you for your contributions!
