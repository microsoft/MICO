import io
import matplotlib
import pandas as pd
import matplotlib.pyplot as plt


def image_to_html(fig):
    """Converts a matplotlib plot to SVG"""
    iostring = io.StringIO()
    fig.savefig(iostring, format="svg", bbox_inches=0, dpi=300)
    iostring.seek(0)

    return iostring.read()


def generate_roc(fpr, tpr):
    fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(8,3.5))

    ax2.semilogx()
    ax2.semilogy()
    ax2.set_xlim(1e-5,1)
    ax2.set_ylim(1e-5,1)
    ax2.set_xlabel("False Positive Rate")
    #ax2.set_ylabel("True Positive Rate")
    ax2.plot([0, 1], [0, 1], ls=':', color='grey')

    ax1.set_xlim(0,1)
    ax1.set_ylim(0,1)
    ax1.set_xlabel("False Positive Rate")
    ax1.set_ylabel("True Positive Rate")
    ax1.plot([0,1], [0,1], ls=':', color='grey')

    ax1.plot(fpr, tpr)
    ax2.plot(fpr, tpr)

    return fig


def generate_table(scores):
    table = pd.DataFrame(scores).T
    table.drop(["fpr", "tpr"], axis=1, inplace=True)
    # replace = {
    #     "inf": "No DP",
    #     "hi":  "High &epsi;",
    #     "lo":  "Low &epsi;",
    # }
    # table.index = [replace[i] for i in table.index]
    replace_column = {
        "accuracy":  "Accuracy",
        "AUC": "AUC-ROC",
        "MIA": "MIA",
        "TPR_FPR_10": "TPR @ 0.001 FPR",
        "TPR_FPR_100": "TPR @ 0.01 FPR",
        "TPR_FPR_500": "TPR @ 0.05 FPR",
        "TPR_FPR_1000": "TPR @ 0.1 FPR",
        "TPR_FPR_1500": "TPR @ 0.15 FPR",
        "TPR_FPR_2000": "TPR @ 0.2 FPR",
    }
    table.columns = [replace_column[c] for c in table.columns]

    return table


def generate_html(scores):
    """Generates the HTML document as a string, containing the various detailed scores"""
    matplotlib.use('Agg')

    img = {}
    for scenario in scores:
        fpr = scores[scenario]["fpr"]
        tpr = scores[scenario]["tpr"]
        fig = generate_roc(fpr, tpr)
        fig.tight_layout(pad=1.0)

        img[scenario] = f"<h2>{scenario}</h2><div>{image_to_html(fig)}</div>"

    table = generate_table(scores)

    # Generate the HTML document.
    css = '''
    body {
        background-color: #ffffff;
    }
    h1 {
        text-align: center;
    }
    h2 {
        text-align: center;
    }
    div {
        white-space: normal;
        text-align: center;
    }
    table {
      border-collapse: collapse;
      margin: auto;
    }
    table > :is(thead, tbody) > tr > :is(th, td) {
      padding: 5px;
    }
    table > thead > tr > :is(th, td) {
      border-top:    2px solid; /* \toprule */
      border-bottom: 1px solid; /* \midrule */
    }
    table > tbody > tr:last-child > :is(th, td) {
      border-bottom: 2px solid; /* \bottomrule */
    }'''

    html = f'''<!DOCTYPE html>
    <html>
    <head>
        <title>MICO - Detailed scores</title>
        <style>
        {css}
        </style>
    </head>
    <body>

    <div>
    {table.to_html(border=0, float_format='{:0.4f}'.format, escape=False)}
    </div>'''

    for scenario in scores:
        html += img[scenario]

    html += "</body></html>"

    return html