# Streamlit Multi-Tab Application

A modular Streamlit application with tab-based architecture designed for parallel development by multiple developers.

## 🏗️ Project Structure

```
MetaOpt_Model_SurgicalForecasting/
│
├── app.py                          # Main application entry point
├── requirements.txt                # Python dependencies
├── tabs/                           # Tab modules directory
│   ├── __init__.py                # Package initialization
│   ├── tab1_home.py               # Home tab (Developer 1)
│   ├── tab2_data_analysis.py     # Data Analysis tab (Developer 2)
│   ├── tab3_visualization.py     # Visualization tab (Developer 3)
│   └── tab4_settings.py          # Settings tab (Developer 4)
│
└── README_STREAMLIT.md            # This file
```

## 🚀 Getting Started

### Installation

1. Install the required dependencies:
```bash
pip install -r requirements.txt
```

2. Run the application:
```bash
streamlit run app.py
```

The app will open in your default browser at `http://localhost:8501`

## 👥 Developer Guide

### Working on Individual Tabs

Each tab is completely independent and can be developed separately:

#### Developer 1 - Home Tab (`tabs/tab_home.py`)
- Welcome page and overview
- Application metrics and statistics
- Getting started information

#### Developer 2 - Data Analysis Tab (`tabs/tab_data_analysis.py`)
- Data upload and processing
- Statistical analysis features
- Data transformation tools

#### Developer 3 - Visualization Tab (`tabs/tab_visualization.py`)
- Chart creation and customization
- Data visualization options
- Interactive plotting features

#### Developer 4 - Settings Tab (`tabs/tab_settings.py`)
- User preferences
- Application configuration
- API settings and integration

### How to Add a New Tab

1. **Create a new file** in the `tabs/` directory (e.g., `tab_reports.py`)

2. **Follow this template**:
```python
"""
Tab 5: Reports Tab

Developer: [Your Name]
Description: Brief description of this tab's purpose
"""

import streamlit as st


def render():
    """
    Render the Reports tab content.
    
    This function is called by the main app when this tab is active.
    Keep all tab logic self-contained within this module.
    """
    st.header("📄 Reports")
    
    # Your tab content here
    st.write("Content for this tab...")


# Add helper functions below
def _helper_function():
    """Private helper function for this tab."""
    pass
```

3. **Import the new tab** in `tabs/__init__.py`:
```python
from . import tab_reports

__all__ = [
    'tab_home',
    'tab_data_analysis',
    'tab_visualization',
    'tab_settings',
    'tab_reports'  # Add your new tab
]
```

4. **Add the tab to the main app** in `app.py`:
```python
from tabs import tab1_home, tab2_data_analysis, tab3_visualization, tab4_settings, tab5_reports

# In the main() function:
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🏠 Home",
    "📈 Data Analysis",
    "📊 Visualization",
    "⚙️ Settings",
    "📄 Reports"  # Add your tab
])

# Add the rendering:
with tab5:
    tab_reports.render()
```

## 🔑 Key Principles

### 1. **Isolation**
Each tab module is self-contained. All imports, functions, and state management should be within the tab file.

### 2. **render() Function**
Every tab must have a `render()` function that serves as the entry point. This is called by `app.py`.

### 3. **Helper Functions**
Use private functions (prefixed with `_`) for internal logic within each tab.

### 4. **Session State**
If you need to share data between tabs, use `st.session_state`:
```python
# Set data in one tab
st.session_state['shared_data'] = my_data

# Access in another tab
if 'shared_data' in st.session_state:
    data = st.session_state['shared_data']
```

### 5. **No Cross-Tab Dependencies**
Tabs should not import from each other. If you need shared functionality, create a separate `utils/` directory.

## 🛠️ Best Practices

1. **Keep tabs focused**: Each tab should have a clear, single purpose
2. **Use docstrings**: Document your functions and their parameters
3. **Handle errors gracefully**: Use try-except blocks for error-prone operations
4. **Add comments**: Explain complex logic for other developers
5. **Test independently**: Each tab should work without requiring other tabs
6. **Use st.cache_data**: Cache expensive computations to improve performance

## 📦 Adding Dependencies

If your tab needs additional Python packages:

1. Add them to `requirements.txt`:
```
your-package>=1.0.0
```

2. Install for your environment:
```bash
pip install -r requirements.txt
```

3. Import in your tab file:
```python
import your_package
```

## 🐛 Troubleshooting

### Import errors
- Make sure you're running the app from the project root directory
- Ensure `__init__.py` exists in the `tabs/` directory

### Tab not showing
- Check that the tab is imported in `app.py`
- Verify the `render()` function exists in your tab file
- Look for any Python syntax errors in your tab file

### Streamlit not found
- Install dependencies: `pip install -r requirements.txt`
- Verify installation: `streamlit --version`

## 📝 Git Workflow for Teams

To avoid conflicts when working on separate tabs:

1. **Pull latest changes**:
```bash
git pull origin main
```

2. **Create a feature branch**:
```bash
git checkout -b feature/tab-your-feature
```

3. **Work on your tab** (only modify your assigned tab file)

4. **Commit your changes**:
```bash
git add tabs/your_tab_file.py
git commit -m "Add: description of your changes"
```

5. **Push and create pull request**:
```bash
git push origin feature/tab-your-feature
```

## 🎨 Customization

### Changing Page Configuration
Edit `app.py` to modify the page config:
```python
st.set_page_config(
    page_title="Your App Name",
    page_icon="🚀",
    layout="wide"  # or "centered"
)
```

### Adding Custom Styling
Create a `.streamlit/config.toml` file for custom themes and styling.

## 📚 Resources

- [Streamlit Documentation](https://docs.streamlit.io/)
- [Streamlit API Reference](https://docs.streamlit.io/library/api-reference)
- [Streamlit Gallery](https://streamlit.io/gallery)

## 🤝 Contributing

Each developer should:
- Focus on their assigned tab
- Follow the coding standards
- Add tests if applicable
- Document any new features
- Create pull requests for review

---

**Happy Coding! 🎉**
