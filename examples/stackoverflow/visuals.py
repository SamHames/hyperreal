"""
This script produces a bunch of visualisations and tabular summaries of the dataset.

"""

from datetime import datetime
import os

from hyperreal import index, corpus, utilities
import leather
import tablib

if __name__ == "__main__":
    try:
        os.mkdir("outputs")
    except FileExistsError:
        pass

    leather.theme.default_chart_height = 400
    leather.theme.default_chart_width = 800
    leather.theme.background_color = "white"

    corp = corpus.StackExchangeCorpus("data/stackoverflow.db")
    idx = index.Index("data/stackoverflow_index.db")

    # Top tags in the dataset
    tag_counts = list(
        idx.db.execute(
            """
        select
            value,
            docs_count
        from inverted_index
        where field='Tag'
        order by docs_count desc limit 15
        """
        )
    )

    ds = tablib.Dataset()

    for row in tag_counts:
        ds.append(row)

    ds.headers = ["Tag", "Post Count"]

    with open("outputs/top_tags.html", "w") as table:
        table.write(ds.export("html"))

    chart = leather.Chart("Top Stackoverflow Tags")
    chart.add_bars(tag_counts, x=1, y=0)
    chart.to_svg("outputs/top_tag_columns.svg")

    # Trends for questions and answers over time.
    query = {"Question": idx[("PostType", "Question")]}

    values, totals, intersections = idx.intersect_queries_with_field(
        query, "created_month"
    )

    dates = [datetime.fromisoformat(date) for date in values]

    chart = leather.Chart("Monthly Posts and Questions")
    question_counts = list(zip(dates, intersections["Question"]))
    total_counts = list(zip(dates, totals))
    chart.add_line(question_counts, name="Questions", width=1)
    chart.add_line(total_counts, name="All Posts", width=3)
    chart.add_y_axis(name="Posts")

    ds = tablib.Dataset()
    ds.append_col(
        dates,
    )
    ds.append_col(intersections["Question"])
    ds.append_col(totals)

    ds.headers = ["Month", "Question Count", "Post Count"]

    with open("outputs/post_trend.html", "w") as table:
        table.write(ds.export("html"))

    chart.to_svg("outputs/overall_trends.svg")

    # Trends for all word clusters over time

    clusters = idx.top_cluster_features(top_k=5)
    queries = {
        " ".join([r[2] for r in features] + ["..."]): idx.cluster_docs(cluster_id)
        for cluster_id, _, features in clusters
    }

    values, totals, intersections = idx.intersect_queries_with_field(
        queries, "created_month"
    )

    leather.theme.default_chart_height = 150
    leather.theme.default_chart_width = 450

    grid = leather.Grid()
    for query, counts in intersections.items():
        chart = leather.Chart(query)
        data = list(zip(dates, [100 * c / t for c, t in zip(counts, totals)]))[1:-1]
        chart.add_line(data)
        chart.add_y_axis(name="% of Posts")
        grid.add_one(chart)

    grid.to_svg("outputs/stackoverflow_word_usage.svg")

    # Trends for different frontend frameworks as validation
    chosen_clusters = [491, 1013, 441]
    queries = {
        " ".join(
            [r[2] for r in idx.cluster_features(cluster_id)[:5]] + ["..."]
        ): idx.cluster_docs(cluster_id)
        for cluster_id in chosen_clusters
    }

    values, totals, intersections = idx.intersect_queries_with_field(
        queries, "created_month"
    )

    leather.theme.default_chart_height = 400
    leather.theme.default_chart_width = 800

    chart = leather.Chart("Javascript and Frameworks")
    for query, counts in intersections.items():
        data = list(zip(dates, [100 * c / t for c, t in zip(counts, totals)]))[1:-1]
        chart.add_line(data, name=query)

    chart.add_y_axis(name="% of All Posts")

    chart.to_svg("outputs/javascript_trends.svg")

    # Trends for different infrastructury things as validation
    # [479, 767] Docker and kubernetes
    chosen_clusters = [971, 893, 373]
    queries = {
        " ".join(
            [r[2] for r in idx.cluster_features(cluster_id)[:5]] + ["..."]
        ): idx.cluster_docs(cluster_id)
        for cluster_id in chosen_clusters
    }

    values, totals, intersections = idx.intersect_queries_with_field(
        queries, "created_month"
    )

    leather.theme.default_chart_height = 400
    leather.theme.default_chart_width = 800

    chart = leather.Chart("Orchestration/Containerisation")
    for query, counts in intersections.items():
        data = list(zip(dates, [100 * c / t for c, t in zip(counts, totals)]))[1:-1]
        chart.add_line(data, name=query)

    chart.add_y_axis(name="% of All Posts")

    chart.to_svg("outputs/infrastructure_trends.svg")

    # Trends for overhyped things
    chosen_clusters = [1, 721, 312]
    queries = {
        " ".join(
            [r[2] for r in idx.cluster_features(cluster_id)[:5]] + ["..."]
        ): idx.cluster_docs(cluster_id)
        for cluster_id in chosen_clusters
    }

    values, totals, intersections = idx.intersect_queries_with_field(
        queries, "created_month"
    )

    leather.theme.default_chart_height = 400
    leather.theme.default_chart_width = 800

    chart = leather.Chart("Overhyped")
    for query, counts in intersections.items():
        data = list(zip(dates, [100 * c / t for c, t in zip(counts, totals)]))[1:-1]
        chart.add_line(data, name=query)

    chart.add_y_axis(name="% of All Posts")

    chart.to_svg("outputs/overhyped_trends.svg")

    # # Trends for word cluster usage relative to the Tag python over time.
    # tag_python = idx[("Tag", "python")] & idx[("PostType", "Question")]
    # clusters = idx.pivot_clusters_by_query(tag_python, top_k=5)

    # queries = {
    #     " ".join([r[2] for r in features] + ["..."]): tag_python
    #     & idx.cluster_docs(cluster_id)
    #     for cluster_id, _, features in clusters
    # }

    # values, totals, intersections = idx.intersect_queries_with_field(
    #     queries, "created_month"
    # )
    # (
    #     python_values,
    #     python_totals,
    #     python_intersections,
    # ) = idx.intersect_queries_with_field({"tag_python": tag_python}, "created_month")

    # dates = [datetime.fromisoformat(date) for date in values]

    # leather.theme.default_chart_height = 150
    # leather.theme.default_chart_width = 450
    # grid = leather.Grid()
    # display_order = sorted(intersections.items(), reverse=True, key=lambda x: sum(x[1]))
    # for query, counts in display_order:
    #     chart = leather.Chart(query)
    #     data = list(
    #         zip(
    #             dates,
    #             [
    #                 100 * c / (t or 1)
    #                 for c, t in zip(counts, python_intersections["tag_python"])
    #             ],
    #         )
    #     )[1:-1]
    #     chart.add_line(data)
    #     chart.add_y_axis(name="% of Questions")
    #     grid.add_one(chart)

    # grid.to_svg("outputs/stackoverflow_python_tag_question_word_usage.svg")

    # # Trends for word cluster usage relative to the Tag python over time.
    # tag_python_answers = idx[("Tag", "python")] - idx[("PostType", "Question")]
    # clusters = idx.pivot_clusters_by_query(tag_python_answers, top_k=5)

    # queries = {
    #     " ".join([r[2] for r in features] + ["..."]): tag_python_answers
    #     & idx.cluster_docs(cluster_id)
    #     for cluster_id, _, features in clusters
    # }

    # values, totals, intersections_answers = idx.intersect_queries_with_field(
    #     queries, "created_month"
    # )
    # (
    #     python_values,
    #     python_totals,
    #     python_intersections,
    # ) = idx.intersect_queries_with_field({"tag_python": tag_python}, "created_month")

    # dates = [datetime.fromisoformat(date) for date in values]

    # leather.theme.default_chart_height = 150
    # leather.theme.default_chart_width = 450
    # grid = leather.Grid()
    # display_order = sorted(intersections.items(), reverse=True, key=lambda x: sum(x[1]))
    # for query, counts in display_order:
    #     chart = leather.Chart(query)
    #     data = list(
    #         zip(
    #             dates,
    #             [
    #                 100 * c / (t or 1)
    #                 for c, t in zip(counts, python_intersections["tag_python"])
    #             ],
    #         )
    #     )[1:-1]
    #     chart.add_line(data)
    #     chart.add_y_axis(name="% of Questions")
    #     grid.add_one(chart)

    # grid.to_svg("outputs/stackoverflow_python_tag_answer_word_usage.svg")

    ## All Python Q+A things over time as individual plots.
    tag_python_q = idx[("Tag", "python")] & idx[("PostType", "Question")]
    tag_python_a = idx[("Tag", "python")] - idx[("PostType", "Question")]
    clusters = list(idx.pivot_clusters_by_query(idx[("Tag", "python")], top_k=5))

    queries_q = {
        " ".join([r[2] for r in features] + ["..."]): tag_python_q
        & idx.cluster_docs(cluster_id)
        for cluster_id, _, features in clusters
    }

    queries_a = {
        " ".join([r[2] for r in features] + ["..."]): tag_python_a
        & idx.cluster_docs(cluster_id)
        for cluster_id, _, features in clusters
    }

    values, totals_q, intersections_q = idx.intersect_queries_with_field(
        queries_q, "created_month"
    )
    values, totals_a, intersections_a = idx.intersect_queries_with_field(
        queries_a, "created_month"
    )

    (
        _,
        _,
        python_intersections,
    ) = idx.intersect_queries_with_field(
        {"tag_python_a": tag_python_a, "tag_python_q": tag_python_q}, "created_month"
    )

    dates = [datetime.fromisoformat(date) for date in values]

    leather.theme.default_chart_height = 400
    leather.theme.default_chart_width = 800

    # Overall amount of Questions and Answers to questions tagged with Python.
    chart = leather.Chart("All Python Tagged Q+A")

    tag_q = list(
        zip(
            dates,
            [
                100 * c / (t or 1)
                for c, t in zip(python_intersections["tag_python_q"], totals_q)
            ],
        )
    )[1:-1]

    tag_a = list(
        zip(
            dates,
            [
                100 * c / (t or 1)
                for c, t in zip(python_intersections["tag_python_a"], totals_a)
            ],
        )
    )[1:-1]

    chart.add_line(tag_q, name="Questions", width=1)
    chart.add_line(tag_a, name="Answers", width=3)
    chart.add_y_axis(name="% of Q/A")
    chart.to_svg("outputs/python_tag_overall_trend.svg")

    # Plot all of the individual trends
    display_order = sorted(
        intersections_q.items(), reverse=True, key=lambda x: sum(x[1])
    )
    for query, counts in display_order:
        chart = leather.Chart(query)
        data_q = list(
            zip(
                dates,
                [
                    100 * c / (t or 1)
                    for c, t in zip(counts, python_intersections["tag_python_q"])
                ],
            )
        )[1:-1]
        chart.add_line(data_q, name="Questions", width=1)
        data_a = list(
            zip(
                dates,
                [
                    100 * c / (t or 1)
                    for c, t in zip(
                        intersections_a[query], python_intersections["tag_python_a"]
                    )
                ],
            )
        )[1:-1]
        chart.add_line(data_a, name="Answers", width=3)
        chart.add_y_axis(name="% of Python Q/A")
        chart.to_svg(f"outputs/{query}.svg")
