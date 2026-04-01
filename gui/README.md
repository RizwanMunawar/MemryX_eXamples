# MemryX Examples Launcher

The **MemryX Examples Launcher** is a browser-based interface for exploring and running applications from the MemryX examples repository.

<picture>
  <img src="assets/gui_gif.gif" alt="MemryX Examples Launcher preview">
</picture>

It brings the most common example workflows into one place, so you can:

- Browse examples by category
- Open example documentation and tutorials quickly
- Launch supported demos from the browser
- Monitor live runtime logs

## Getting Started

Choose the installation flow that matches your setup.

### Option 1: Install from a local checkout

Use this option if you already have the repository cloned locally.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --extra-index-url https://developer.memryx.com/pip -e ".[memryx-sdk]"
example_launcher
```

### Option 2: Install from GitHub

Use this option to install the launcher directly from the repository.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --extra-index-url https://developer.memryx.com/pip -e "git+ssh://git@github.com/memryx/memryx_examples_internal.git@gui_app#egg=memryx-examples-launcher[memryx-sdk]"
example_launcher
```

> [!NOTE]
> The launcher starts a local web server and automatically opens the application in your default browser at `http://localhost:8080`.

## Typical Workflow

1. Activate the virtual environment:

   ```bash
   source .venv/bin/activate
   ```

2. Start the launcher:

   ```bash
   example_launcher
   ```

3. In the browser, select an example and click **Run**
4. Monitor the output in the live panel
5. Stop the launcher with `Ctrl+C`

## What the Launcher Provides

The launcher helps you:

- Explore available examples from a single dashboard
- Navigate examples by category
- Access example documentation and tutorials quickly
- Launch supported applications without manually entering each directory
- Watch runtime logs as the example executes

## Troubleshooting

### Example fails after clicking Run

The launcher can start an example, but each example may still require its own:

- model files
- assets
- dependencies
- environment setup

If an example fails to run, open that example’s README and verify that all prerequisites have been completed.

### Browser does not open automatically

If the launcher does not open a browser window automatically, open the following URL manually:

```text
http://localhost:8080
```

## License

This application is part of the MemryX examples repository and is distributed under the [LGPL](LICENSE.md).

### Third-Party Components

This launcher may depend on third-party Python packages and other open-source components. Please refer to those projects for their respective license terms.
