"""Console prompts for InstaCrawl control."""

from rich.layout import Layout
from rich.panel import Panel
from rich.prompt import Confirm, Prompt, IntPrompt
from instacrawl.insta import InstagramData
from typing import List, Dict, Optional
import pandas as pd
import dtale
from os import PathLike
from pathlib import Path
import os
import time
from datetime import datetime
from instacrawl.cli_utils import align_df, save_df, read_df
from instacrawl.analysis import Postalyzer
from dotenv import dotenv_values

empty_entry = ["", "\n", "\r", " "]


class InstaCrawl:
    def __init__(self, console, **kwargs):
        self.console = console
        self.config: Dict[str, Optional[str]]
        if os.path.exists(os.path.expanduser("~/.instacrawlrc")):
            self.config = {
                **dotenv_values(os.path.expanduser("~/.instacrawlrc")),
                **os.environ,
                **kwargs,
            }
        else:
            self.config = {
                **os.environ,
                **kwargs,
            }
        self.welcome()

    def print(self, text: str, input: str):
        """Print the standard format."""
        self.console.clear()
        layout = Layout(size=20)
        layout.split_column(
            Layout(
                text,
                name="text",
                ratio=1
            ),
            Layout(
                input,
                name="input",
                ratio=1
            )
        )
        panel = Panel(
            layout,
            expand=False,
            highlight=True,
            width=80,
            height=20,
            border_style="white",
            title="[bold green]InstaCrawl[/bold green]"
        )
        self.console.print(panel)

    def welcome(self):
        """Display the welcome screen and get the user's choice."""
        self.print(
            """
Welcome to [bold]InstaCrawl[/bold], the Instagram performance analyzer!

[bold]InstaCrawl[/bold] is a CLI tool that allows you to crawl Instagram
profiles, analyze their performance, and download their posts.
            """,
            "Press [bold]Y/n[/bold] to continue."
        )
        if Confirm.ask(console=self.console):
            self.options()
        else:
            return exit()

    def options(self):
        """Display the options screen and get the user's choice."""
        self.print(
            "What would you like to do?",
            """
Press [bold]1[/bold] to start crawling a new profile
Press [bold]2[/bold] to scrape saved data from a crawl
Press [bold]3[/bold] to download posts from a crawl
Press [bold]4[/bold] to align a crawl
Press [bold]5[/bold] to analyze a crawl
Press [bold]6[/bold] to view the results from a crawl
Press [bold]0[/bold] to exit
            """
        )
        match IntPrompt.ask("", console=self.console):
            case 1:
                self.step1()
            case 2:
                self.step2()
            case 3:
                self.step3()
            case 4:
                self.step4()
            case 5:
                self.step5()
            case 6:
                self.step6()
            case _:  # noqa: F811
                return exit()

    def get_new_data(self) -> tuple[str, PathLike, str, str]:
        """Get the default data from the user."""
        self.print(
            "What is the username that you'd like to crawl?",
            """
This should be without the @ symbol.
Alternatively, you can paste the URL of the profile here.
            """
        )
        url = Prompt.ask("Profile", console=self.console)
        if not url.startswith("https://instagram.com/"):
            url = "https://instagram.com/" + url

        if ("INSTACRAWL_SAVE_TO" in self.config.keys() and
                self.config["INSTACRAWL_SAVE_TO"] not in empty_entry and
                self.config["INSTACRAWL_SAVE_TO"] is not None):
            save_to = Path(
                os.path.expanduser(
                    str(
                        self.config["INSTACRAWL_SAVE_TO"]
                    )
                )
            )
        else:
            self.print(
                "Where would you like to save the data?",
                """
This should be a directory. If it does not exist, it will be created.

This is where all of the generated data will be stored. This will allow you
to analyze the data later without having to crawl the profile again.

[italic](Default is ~/instacrawl)[/italic]

[bold blue]NOTE:[/bold blue] This will be saved in an [italic bold]~/.instacrawlrc[/italic bold] file.
If you would like to change this, you can edit or simply delete that file.
It will be recreated if you start a new crawl.
                """
            )
            save_to = Prompt.ask("Path to save data", console=self.console)
            if save_to in empty_entry:
                save_to = Path(os.path.expanduser("~/instacrawl"))
            else:
                save_to = Path(os.path.expanduser(save_to))
            with open(os.path.expanduser("~/.instacrawlrc"), "a") as f:
                f.writelines(f"INSTACRAWL_SAVE_TO={str(save_to)}\n")

        if ("INSTACRAWL_USERNAME" in self.config.keys() and
                self.config["INSTACRAWL_USERNAME"] not in empty_entry and
                self.config["INSTACRAWL_USERNAME"] is not None):
            username = str(
                        self.config["INSTACRAWL_USERNAME"]
                    )
        else:
            self.print(
                "What is your Instagram username?",
                """
This is used to log into your account as Instagram requires you to be
logged in to view more than 9 posts. This is not stored anywhere.

[italic]This can also be an email or phone number.[/italic]

[bold green]NOTE:[/bold green] If you have two-factor authentication enabled,
you will need to disable it for this to work. Sometimes this is not the case
if you've logged in recently on the same network.

[bold]
IT IS RECOMMENDED THAT THIS USERNAME IS DIFFERENT THAN THE ONE YOU ARE CRAWLING
[/bold]

[bold blue]NOTE:[/bold blue] This will be saved in an [italic bold]~/.instacrawlrc[/italic bold] file.
If you would like to change this, you can edit or simply delete that file.
It will be recreated if you start a new crawl.
                """
            )
            username = Prompt.ask(
                "Username",
                console=self.console
            )
            with open(os.path.expanduser("~/.instacrawlrc"), "a") as f:
                f.writelines(f"INSTACRAWL_USERNAME={str(username)}\n")

        if ("INSTACRAWL_PASSWORD" in self.config.keys() and
                self.config["INSTACRAWL_PASSWORD"] not in empty_entry and
                self.config["INSTACRAWL_PASSWORD"] is not None):
            password = str(
                        self.config["INSTACRAWL_PASSWORD"]
                    )
        else:
            self.print(
                "What is your Instagram password?",
                """
This is used to log into your account as Instagram requires you to be
logged in to view more than 9 posts. This is not stored anywhere.

[bold green]NOTE:[/bold green] If you have two-factor authentication enabled,
you will need to disable it for this to work. Sometimes this is not the case
if you've logged in recently on the same network.

[bold blue]NOTE:[/bold blue] This will be saved in an [italic bold]~/.instacrawlrc[/italic bold] file.
If you would like to change this, you can edit or simply delete that file.
It will be recreated if you start a new crawl.
                """
            )
            password = Prompt.ask(
                "Password",
                password=True,
                console=self.console
            )
            with open(os.path.expanduser("~/.instacrawlrc"), "a") as f:
                f.writelines(f"INSTACRAWL_PASSWORD={str(password)}\n")
        return url, save_to, username, password

    def get_old_data(self) -> PathLike:
        """Get the default data from the user."""
        if ("INSTACRAWL_SAVE_TO" in self.config.keys() and
                self.config["INSTACRAWL_SAVE_TO"] not in empty_entry and
                self.config["INSTACRAWL_SAVE_TO"] is not None):
            save_to = Path(
                os.path.expanduser(
                    str(
                        self.config["INSTACRAWL_SAVE_TO"]
                    )
                )
            )
        else:
            self.print(
                "Where is the data saved?",
                """
This should be the path to the folder that contains the data
from the previous steps. (e.g. ~/instacrawl for the default,
with the data saved in ~/instacrawl/step_one.pkl, etc.)
                """
            )
            save_to = Prompt.ask("Path to data", console=self.console)
            if save_to == "\n" or save_to == "":
                save_to = Path(os.path.expanduser("~/instacrawl"))
            else:
                save_to = Path(os.path.expanduser(save_to))
        return save_to

    def step1(self, data: InstagramData | None = None):
        """Display the step 1 screen."""
        if data is None:
            url, save_to, username, password = self.get_new_data()
            self.print(
                "Perfect! We're all set to start crawling!",
                """
Once we begin crawling, you can press
[bold]CTRL + C[/bold] to stop at any time.
                """
            )
            time.sleep(3)
            self.print(
                """
While crawling Google Chrome may open multiple times. This is normal.

[bold]NOTE:[/bold] If you are using a Mac, you may need to allow InstaCrawl
to control your computer in order for the crawling to work. You can do this in
[bold blue]
System Preferences > Security & Privacy > Privacy > Accessibility
[/bold blue]

[bold]DISCLAIMER:[/bold] Crawling and scraping is against Instagram's Terms
of Service and may result in the profile you provided a username and password
for being deactivated. However, this is rare, and in most cases Instagram
will only temporarily log you out for an hour or so. If you are concerned
about this, you can create a new Instagram account and use that instead.

[green]InstaCrawler can take [italic]multiple hours[/italic] to complete
in order to get around Instagram's attemps to block crawling.

Please make sure your computer is plugged in and set to not
[bold]sleep / lock / log out / hibernate[/bold].[/green]
                """,
                "Would you like to begin crawling?"
            )
            if not Confirm.ask(console=self.console):
                return exit()
            data = InstagramData()
            data.new(
                url,
                save_to,
                username,
                password
            )
            self.console.clear()
            date = self.get_date()
            if date is None or date in ["", "\n", "\r", " "]:
                data.run()
            else:
                assert isinstance(date, datetime)
                data.run_until(date)
            self.print(
                "Step 1 complete!",
                """
Now you will notice that the save data directory has a new file
named [bold]step_one.pkl[/bold]. This file contains all of the data
from step 1 so that you can skip it in the future.

Would you like to continue on to [bold green]Step 2[/bold green]?
                """
            )
            if not Confirm.ask(console=self.console):
                return exit()
            self.step2(data)

    def step2(self, data: InstagramData | None = None):
        """Display the step 2 screen."""
        if data is None:
            save_to = self.get_old_data()
            data = InstagramData(
                save_to,
            )
            data.load_step_one()
        data.analyze()
        self.print(
            "Step 2 complete!",
            """
Now you will notice that the save data directory has a new file
named [bold]step_two.pkl[/bold]. This file contains all of the data
from step 1 and 2 so that you can skip it in the future.

Would you like to continue on to [bold green]Step 3[/bold green]?
            """
        )
        if not Confirm.ask(console=self.console):
            return exit()
        self.step3(data)

    def step3(self, data: InstagramData | None = None):
        """Display the step 3 screen."""
        if data is None:
            save_to = self.get_old_data()
            data = InstagramData(
                save_to,
            )
            data.load_step_one()
            data.load_step_two()
        data.download()
        self.print(
            "Step 3 complete!",
            """
Now you will notice that the save data directory has a new file
named [bold]step_three.pkl[/bold]. This file contains all of the data
from step 1 through 3 so that you can skip it in the future.

Would you like to continue on to [bold green]Step 4[/bold green]?
            """
        )
        if not Confirm.ask(console=self.console):
            return exit()
        return self.step4(data)

    def step4(
        self,
        data: InstagramData | None = None,
    ):
        """Display the step 4 screen."""
        if data is None:
            save_to = self.get_old_data()
            data = InstagramData(
                save_to,
            )
            data.load_step_one()
            data.load_step_two()
            data.load_step_three()
        df = data.generate()
        align_df(df)
        self.print(
            """
[bold]Where would you like to save the data sheet?[/bold]
            """,
            """
This can be in any format that Pandas supports.
Examples include .csv, .xlsx, .json, etc.

[italic](Default: ~/instacrawl/data.csv)[/italic]
            """)
        path = Prompt.ask(
            "Saved data path",
            console=self.console
        )
        if path == "":
            path = "~/instacrawl/data.csv"
        path = os.path.expanduser(path)
        save_df(df, path)
        data.save_step_four()
        self.print(
            "Step 4 complete!",
            f"""
Now you will notice that the save data directory has a new file
named [bold]{path}[/bold]. This file contains all of the data
from step 1 through 4 so that you can skip it in the future.

Would you like to continue on to [bold green]Step 5[/bold green]?
            """
        )
        if not Confirm.ask(console=self.console):
            return exit()
        return self.step5(data)

    def step5(
        self,
        data: InstagramData | None = None
    ):
        if data is None:
            save_to = self.get_old_data()
            data = InstagramData(
                save_to
            )
            data.load_step_one()
            data.load_step_two()
            data.load_step_three()
            data.load_step_four()
        analysis = Postalyzer(data.step_four)
        df: List[pd.DataFrame] = analysis.analyze()
        df.append(data.generate(multi=True))
        final = pd.concat(df, axis=1)
        final.to_pickle(os.path.join(data.path, "step_five.pkl"))
        self.print(
            "Step 5 complete!",
            """
Now you will notice that the save data directory has a new file
named [bold]~/instacrawl/data.csv[/bold]. This file contains all of the data
from step 1 through 5 so that you can skip it in the future.

Would you like to continue on to [bold green]Step 6[/bold green]?
            """
        )
        if not Confirm.ask(console=self.console):
            return exit()
        return self.step6(final)

    def step6(
        self,
        df: pd.DataFrame | None = None,
    ):
        if df is None:
            save_to = self.get_old_data()
            df = read_df(os.path.join(save_to, "step_five.pkl"))
        dtale.show(df, subprocess=False)

    def get_date(self) -> str | datetime:
        """Get the date to crawl until."""
        self.print(
            """
[bold]What date would you like to crawl until?[/bold]
[italic]Leave empty to crawl the first 21 posts[/italic]
            """,
            """
(Format is YYYY-MM-DD)
            """
        )
        date = Prompt.ask(
            "Date",
            console=self.console
        )
        if date == "":
            return date
        else:
            try:
                date = datetime.fromisoformat(date)
                return date
            except ValueError:
                self.print(
                    """
[bold red]
Invalid date format. Please try again.
[/bold red]
                    """,
                    """
The format is YYYY-MM-DD.
For example, 2021-01-31 for January 31st, 2021.
                    """
                )
                time.sleep(3)
                return self.get_date()
