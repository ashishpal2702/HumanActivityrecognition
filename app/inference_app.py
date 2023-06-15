import gradio as gr

def predict(df):

    return df

demo = gr.Interface(
    predict,
    [
        gr.Textbox(
            label="tBodyAcc-mean()-X",
            info="0.288",
            lines=3,
            value="Provide value between 0 to 1",
        ),
        gr.Textbox(
            label="Text 2",
            info="Text to compare",
            lines=3,
            value="The fast brown fox jumps over lazy dogs.",
        ),
    ],
    gr.HighlightedText(
        label="Diff",
        combine_adjacent=True,
        show_legend=True,
    ).style(color_map={"+": "red", "-": "green"}),
    theme=gr.themes.Base()
)
if __name__ == "__main__":
    demo.launch()
