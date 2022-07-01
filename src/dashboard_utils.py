import ipywidgets
from ipywidgets import GridBox, Layout
from IPython.display import display


def two_columns(
    col1_content=None,
    col2_content=None,
    also_return_outputs=False,
    col1_layout={"border": "1px dotted gray"},
    col2_layout={"border": "1px dotted gray"},
    grid_template_columns="50% 50%",
    grid_template_rows="auto auto",
    width="100%",
):
    col1 = ipywidgets.Output(layout=col1_layout)
    col2 = ipywidgets.Output(layout=col2_layout)
    grid_box = GridBox(
        children=[col1, col2],
        layout=Layout(
            width=width,
            grid_template_rows=grid_template_rows,
            grid_template_columns=grid_template_columns,
            grid_template_areas="""
            "col1 col2"
            """,
        ),
    )

    if col1_content is not None:
        with col1:
            display(col1_content)
    if col2_content is not None:
        with col2:
            display(col2_content)

    if also_return_outputs:
        return grid_box, col1, col2
    else:
        return grid_box
