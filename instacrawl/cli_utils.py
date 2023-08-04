"""CLI utilities functions for printing complex objects."""

from PIL import Image
import pandas as pd
from pynput import keyboard
import cv2
import os
from rich.layout import Layout
from rich.panel import Panel
from rich.text import Text
from rich.live import Live
from rich.table import Table
from instacrawl.console import console
from instacrawl.steps import InstaStepFour


def view_data(data: InstaStepFour):
    """Align the dataframe to the console."""
    console.clear()
    table = Table(
        "Post",
        "Date",
        "Time",
        "Likes",
        "Comments",
        "Link",
        title="Analyzed Data"
    )

    for i in range(0, 15):
        table.add_row(
            f'{i}',
            f'{data.posts[i].date}',
            f'{data.posts[i].time}',
            f'{data.posts[i].likes}',
            f'{data.posts[i].comments}',
            f'{data.posts[i].href}',
        )

    console.print(table)

    console.print("Press 'q' to quit.")
    with keyboard.Events() as events:
        for event in events:
            if event.key == keyboard.KeyCode.from_char("q"):
                return


def align_df(df: pd.DataFrame) -> None:
    """Align the dataframe to the console."""
    i, pixels = print_row(df, 0)
    layout = Layout()
    layout.split_row(
        Layout(pixels, name="image", ratio=2),
        Layout(name="post", ratio=1)
    )
    ins = "Use the up and down arrows or W and S \
to align the image with the post. \
Press SPACE to select the current post."
    layout["post"].split_column(
        Layout(
            Panel(
                Text(
                    ins,
                    justify="center"
                ),
                title="Instructions"
            ),
            name="instructions",
            ratio=1
        ),
        Layout(
            Panel(
                Text(
                    f'{df.iloc[i].loc["article"]}',
                    justify="left",
                    overflow="ellipsis",
                    no_wrap=False
                ),
                title="Caption"
            ),
            name="caption",
            ratio=4
        ),
        Layout(
            Panel(
                Text(
                    f'{df.iloc[i].loc["date"]}',
                    justify="left",
                    overflow="ellipsis",
                ),
                title="Date",
            ),
            name="date",
            ratio=1
        ),
        Layout(
            Panel(
                Text(
                    f'{df.iloc[i].loc["time"]}',
                    justify="left",
                    overflow="ellipsis",
                ),
                title="Time",
            ),
            name="time",
            ratio=1
        ),
        Layout(
            Panel(
                Text(
                    f'{df.iloc[i].loc["likes"]}',
                    justify="left",
                    overflow="ellipsis",
                ),
                title="Likes",
            ),
            name="likes",
            ratio=1
        ),
        Layout(
            Panel(
                Text(
                    f'{df.iloc[i].loc["comments"]}',
                    justify="left",
                    overflow="ellipsis",
                ),
                title="Comments",
            ),
            name="comments",
            ratio=1
        ),
    )

    with Live(layout, refresh_per_second=4, screen=True):
        with keyboard.Events() as events:
            for event in events:
                if (event.key is keyboard.KeyCode.from_char("w")
                        or event.key is keyboard.KeyCode.from_char("W")
                        or event.key is keyboard.Key.up):
                    i = i + 1
                    update_layout(layout, df, i)
                elif (event.key is keyboard.KeyCode.from_char("s")
                        or event.key is keyboard.KeyCode.from_char("S")
                        or event.key is keyboard.Key.down):
                    i = i - 1 if i > 0 else 0
                    update_layout(layout, df, i)
                elif (event.key is keyboard.Key.enter
                        or event.key is keyboard.Key.space):
                    return
        return


def update_layout(layout: Layout, df: pd.DataFrame, i: int):
    """Update the layout with the new post."""
    i, pixels = print_row(df, i)
    layout["image"].update(
        pixels
    )
    layout["post"]["caption"].update(
        Panel(
            Text(
                f'{df.loc[i, "article"]}',
                justify="left",
                overflow="ellipsis",
                no_wrap=False
            ),
            title="Caption"
        )
    )
    layout["post"]["date"].update(
        Panel(
            Text(
                f'{df.loc[i, "date"]}',
                justify="left",
                overflow="ellipsis",
            ),
            title="Date",
        )
    )
    layout["post"]["time"].update(
        Panel(
            Text(
                f'{df.loc[i, "time"]}',
                justify="left",
                overflow="ellipsis",
            ),
            title="Time",
        )
    )
    layout["post"]["likes"].update(
        Panel(
            Text(
                f'{df.loc[i, "likes"]}',
                justify="left",
                overflow="ellipsis",
            ),
            title="Likes",
        )
    )
    layout["post"]["comments"].update(
        Panel(
            Text(
                f'{df.loc[i, "comments"]}',
                justify="left",
                overflow="ellipsis",
            ),
            title="Comments",
        )
    )


def to_string(
    img: Image.Image,
    dest_height: int,
    dest_width: int,
    unicode: bool = True
) -> str:
    """Convert an image to a string."""
    dest_height = dest_height * 2
    dest_width = dest_width * 2
    img_width, img_height = img.size
    scale_height = img_height / dest_height
    scale_width = img_width / dest_width
    if scale_height > scale_width:
        scale = scale_height
        if scale <= 1:
            scale = 1
        dest_width = int(img_width / scale)
        dest_width = dest_width + 1 if dest_width % 2 != 0 else dest_width
    else:
        scale = scale_width
        if scale <= 1:
            scale = 1
        dest_height = int(img_height / scale)
        dest_height = dest_height + 1 if dest_height % 2 != 0 else dest_height
    img = img.resize((dest_width, dest_height))
    img = img.convert("RGB")
    output = ""

    for y in range(0, dest_height, 2):
        for x in range(dest_width):
            if unicode:
                r1, g1, b1 = img.getpixel((x, y))
                r2, g2, b2 = img.getpixel((x, y + 1))
                output = \
                    output + \
                    f"[rgb({r1},{g1},{b1}) on rgb({r2},{g2},{b2})]▀[/]"
            else:
                r, g, b = img.getpixel((x, y))
                output = output + f"[on rgb({r},{g},{b})] [/]"

        output = output + "\n"

    return output


def bw_string(img: Image.Image, dest_height: int, unicode: bool = True) -> str:
    """Convert an image to a string."""
    dest_height = dest_height * 2
    img_width, img_height = img.size
    scale = img_height / dest_height
    dest_width = int(img_width / scale)
    dest_width = dest_width + 1 if dest_width % 2 != 0 else dest_width
    img = img.resize((dest_width, dest_height))
    output = ""

    for y in range(0, dest_height, 2):
        for x in range(dest_width):
            if unicode:
                bw = img.getpixel((x, y))
                bw = img.getpixel((x, y + 1))
                output = \
                    output + \
                    f"[rgb({bw},{bw},{bw}) on rgb({bw},{bw},{bw})]▀[/]"
            else:
                bw = img.getpixel((x, y))
                output = output + f"[on rgb({bw},{bw},{bw})] [/]"

        output = output + "\n"

    return output


def print_row(
        df: pd.DataFrame,
        index: int
) -> tuple[int, str]:
    """Print the row to the console."""
    row = df.loc[index]
    if isinstance(row["paths"], str):
        paths = row["paths"].strip("][").replace("'", "").split(', ')
    elif isinstance(row["paths"], list):
        paths = row["paths"]
        print(paths)
    else:
        raise ValueError("Path is not a string or list.")
    print(row["type_"])
    if row["type_"] == "image":
        print("opening image.")
        with Image.open(str(paths[0])) as image:
            pixels = to_string(
                image,
                console.size.height - 2,
                int(console.size.width * (2/3) - 2),
                True
            )
            return index, pixels
    elif row["type_"] == "video":
        cap = cv2.VideoCapture(str(paths[0]))
        if not cap.isOpened():
            console.print(
                f"""[red]Error opening video stream or file[/red]:
                [bold]{row["paths"][0]}[/bold]"""
            )
        success, image = cap.read()
        if success:
            color_coverted = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(color_coverted)
            pixels = to_string(
                pil_image,
                console.size.height - 2,
                int(console.size.width * (2/3) - 2),
                True
            )
            cap.release()
            return index, pixels
        else:
            console.print(
                f"""[red]Error opening video stream or file[/red]:
                [bold]{row["paths"][0]}[/bold]"""
            )
            return print_row(df, index + 1)
    else:
        return exit()


def save_df(df: pd.DataFrame, path: str):
    """Save the dataframe to a file."""
    extension = path.rsplit(".", 1)[-1]
    file_path = path.rsplit("/", 1)[0]
    if not os.path.exists(file_path):
        os.makedirs(file_path)
    match extension:
        case "csv":
            df.to_csv(path)
        case "json":
            df.to_json(path)
        case "xlsx":
            df.to_excel(path)
        case "orc":
            df.to_orc(path)
        case "parquet":
            df.to_parquet(path)
        case "pickle":
            df.to_pickle(path)
        case "pkl":
            df.to_pickle(path)
        case "hdf":
            df.to_hdf(path, key="instacrawl")
        case "h5":
            df.to_hdf(path, key="instacrawl")
        case "hdf5":
            df.to_hdf(path, key="instacrawl")
        case "feather":
            df.to_feather(path)
        case "ftr":
            df.to_feather(path)
        case "html":
            df.to_html(path)
        case "latex":
            df.to_latex(path)
        case "tex":
            df.to_latex(path)
        case "dta":
            df.to_stata(path)
        case "stata":
            df.to_stata(path)
        case "md":
            df.to_markdown(path)
        case _:
            df.to_clipboard()
            console.print("No path provided, copied to clipboard")


def read_df(path: str) -> pd.DataFrame:
    """Save the dataframe to a file."""
    extension = path.rsplit(".", 1)[-1]
    file_path = path.rsplit("/", 1)[0]
    if not os.path.exists(file_path):
        os.makedirs(file_path)
    match extension:
        case "csv":
            return pd.read_csv(path)
        case "json":
            return pd.read_json(path)
        case "xlsx":
            return pd.read_excel(path)
        case "orc":
            return pd.read_orc(path)
        case "parquet":
            return pd.read_parquet(path)
        case "pickle":
            return pd.read_pickle(path)
        case "pkl":
            return pd.read_pickle(path)
        case "hdf":
            return pd.DataFrame(pd.read_hdf(path, key="instacrawl"))
        case "h5":
            return pd.DataFrame(pd.read_hdf(path, key="instacrawl"))
        case "hdf5":
            return pd.DataFrame(pd.read_hdf(path, key="instacrawl"))
        case "feather":
            return pd.read_feather(path)
        case "ftr":
            return pd.read_feather(path)
        case "html":
            return pd.read_html(path)[0]
        case "dta":
            return pd.DataFrame(pd.read_stata(path))
        case "stata":
            return pd.DataFrame(pd.read_stata(path))
        case _:
            return pd.read_clipboard()
