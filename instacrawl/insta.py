"""Module to scrape data from Instagram profile page."""

from os import PathLike
from pathlib import Path
from typing import List, Dict, Tuple, Literal, Any
from seleniumwire import webdriver
from seleniumwire.utils import decode
from selenium.common.exceptions import TimeoutException
from selenium.webdriver.chrome.options import Options
from selenium.webdriver import ActionChains
from selenium.webdriver.common.by import By
from selenium.webdriver.support.wait import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from concurrent.futures import ThreadPoolExecutor, wait
from requests.utils import cookiejar_from_dict
from rich.progress import Progress
from rich.progress import open as read
from bs4 import BeautifulSoup, NavigableString
from instacrawl.steps import \
    InstaStepOne, \
    InstaStepTwo, \
    InstaStepThree, \
    DownloadRequest, \
    Post, \
    InstaStepFour, \
    InstaPost, \
    PostInfo
from datetime import datetime
from zoneinfo import ZoneInfo
from instacrawl.console import console
from rich.prompt import Prompt
import pandas as pd
import requests
import pickle
import json
import time
import sys
import re
import os


class Cookies(Dict):
    """Class to store cookies."""

    _loading = False

    def __init__(self, cookies: Dict, count: int = 0):
        """Initialize Cookies class."""
        self.count = count
        super().__init__(cookies)

    def refresh(self, link: str = "https://instagram.com"):
        """Refresh cookies."""
        if not self._loading:
            self._loading = True
            options = Options()
            options.page_load_strategy = 'eager'
            driver = webdriver.Chrome(options=options)
            driver.get(link)
            WebDriverWait(driver, 10).until(
                EC.presence_of_element_located(
                    (
                        By.XPATH,
                        "//main[contains(@role, 'main')]"
                    )
                )
            )
            driver.delete_all_cookies()
            driver.get(link)
            WebDriverWait(driver, 10).until(
                EC.presence_of_element_located(
                    (
                        By.XPATH,
                        "//main[contains(@role, 'main')]"
                    )
                )
            )
            selenium_cookies = driver.get_cookies()
            self.clear()
            for cookie in selenium_cookies:
                self[cookie['name']] = cookie['value']
            self.count += 1
            self._loading = False
            driver.close()
        else:
            while self._loading:
                time.sleep(1)

    def refresh_count(self):
        """Return number of times cookies have been refreshed."""
        return self.count


class InstagramData:
    """Class to scrape data from Instagram profile page."""

    def __init__(
        self,
        save_path: PathLike | None = None,
    ):
        """Initialize InstagramData class."""
        self.profile_page: str = ""
        self.username: str = ""
        self.password: str = ""
        if save_path is None:
            self.path: PathLike = Path(os.path.expanduser("~/instacrawl"))
        else:
            self.path: PathLike = Path(os.path.expanduser(save_path))
        self.posts: Dict[
            int, PostInfo
        ] = {}
        self.requests: List[str] = []
        self.articles: List[str] = []
        self.dates: List[datetime] = []
        self.hrefs: List[str] = []

    def new(
        self,
        profile_url: str,
        save_path: PathLike | None,
        username: str,
        password: str
    ):
        self.profile_page = profile_url
        self.username = username
        self.password = password
        if save_path is None:
            self.path: PathLike = Path(os.path.expanduser("~/instacrawl"))
        else:
            self.path: PathLike = Path(os.path.expanduser(save_path))
        if not os.path.exists(self.path):
            os.makedirs(self.path)

    def open(self):
        options = Options()
        options.page_load_strategy = 'eager'
        self.driver = webdriver.Chrome(options=options)
        self.window = self.driver.current_window_handle
        self.req = requests.Session()

    def close(self):
        self.driver.close()
        self.req.close()
        del self.driver
        del self.req
        del self.window

    def login(self):
        """Login to Instagram."""
        with console.status("Logging in..."):
            self.driver.get(self.profile_page)
            try:
                WebDriverWait(self.driver, 10).until(
                    EC.presence_of_element_located(
                        (
                            By.XPATH,
                            "//a[contains(@href, '/accounts/login/?')]"
                        )
                    )
                )

                ActionChains(self.driver).click(
                    self.driver.find_element(
                        By.XPATH,
                        "//a[contains(@href, '/accounts/login/?')]"
                    )
                ).perform()
            except TimeoutException:
                time.sleep(1)

            path = "//*[@id='loginForm']//input[@name='username']"
            WebDriverWait(self.driver, 10).until(
                EC.presence_of_element_located(
                    (
                        By.XPATH,
                        path
                    )
                )
            )
            time.sleep(1)
            ActionChains(self.driver).click(
                self.driver.find_element(
                    By.XPATH,
                    "//*[@id='loginForm']"
                )
            ).perform()
            path = "//*[@id='loginForm']//input[@name='username']"
            ActionChains(self.driver).send_keys_to_element(
                self.driver.find_element(
                    By.XPATH,
                    path
                ),
                self.username
            ).perform()
            path = "//*[@id='loginForm']//input[@name='password']"
            ActionChains(self.driver).send_keys_to_element(
                self.driver.find_element(
                    By.XPATH,
                    path
                ),
                self.password
            ).perform()
            ActionChains(self.driver).click(
                self.driver.find_element(
                    By.XPATH,
                    "//button[@type='submit']"
                )
            ).perform()

            time.sleep(5)

            try:
                WebDriverWait(self.driver, 10).until(
                    EC.presence_of_element_located(
                        (
                            By.XPATH,
                            "//button[text()='Save Info']"
                        )
                    )
                )

                time.sleep(1)

                ActionChains(self.driver).click(
                    self.driver.find_element(
                        By.XPATH,
                        "//section/main/div/div/div/section"
                    )
                ).perform()
                ActionChains(self.driver).click(
                    self.driver.find_element(
                        By.XPATH,
                        "//button[text()='Save Info']"
                    )
                ).perform()
            except TimeoutException:
                time.sleep(1)
            path = "//section/main//article/div[1]/div/div[1]/div[1]/a/div[1]"
            WebDriverWait(self.driver, 20).until(
                EC.presence_of_element_located(
                    (
                        By.XPATH,
                        path
                    )
                )
            )

    def get_login_data(self) -> int:
        """Get data from login page."""
        indicies = []
        with console.status("Getting login data..."):
            soup = BeautifulSoup(self.driver.page_source, "html.parser")
            soup_data = soup.find_all(
                "script",
                type="application/ld+json"
            )
            if isinstance(soup_data, NavigableString) or not soup_data:
                return 0
            else:
                data = json.loads(str(soup_data[1].contents[0]))
                if not data:
                    return 0
                else:
                    for item in data:
                        index = len(list(self.posts.keys()))
                        if index not in self.posts.keys():
                            self.posts[index] = {}
                        indicies.append(index)
                        self.posts[index]["item"] = item
                        dt = datetime.fromisoformat(item.get('dateCreated'))
                        self.dates.append(dt)
                        self.posts[index]["date"] = \
                            f"{dt.year:04d}-{dt.month:02d}-{dt.day:02d}"
                        if dt.hour > 12:
                            if dt.hour == 12:
                                self.posts[index]["time"] = \
                                    f"{dt.hour:02d}:{dt.minute:02d} PM"
                            else:
                                self.posts[index]["time"] = \
                                    f"{(dt.hour - 12):02d}:{dt.minute:02d} PM"
                        elif dt.hour == 0:
                            self.posts[index]["time"] = \
                                f"{12:02d}:{dt.minute:02d} AM"
                        else:
                            self.posts[index]["time"] = \
                                f"{dt.hour:02d}:{dt.minute:02d} AM"
                        self.posts[index]["article"] = \
                            item.get('articleBody')
                        self.articles.append(item.get('articleBody'))
                        for stat in item.get("interactionStatistic"):
                            if stat.get(
                                "interactionType"
                            ) == 'http://schema.org/LikeAction':
                                self.posts[index]["likes"] = \
                                    stat.get(
                                        "userInteractionCount"
                                    )
                            elif stat.get(
                                "interactionType"
                            ) == 'https://schema.org/CommentAction':
                                self.posts[index]["comments"] = \
                                    stat.get(
                                        "userInteractionCount"
                                    )
                        self.posts[index]["content"] = \
                            item.get(
                                'mainEntityOfPage'
                            ).get(
                                '@id'
                            )
                        self.posts[index]["shortcode"] = \
                            item.get(
                                'identifier'
                            ).get(
                                'value'
                            )
        return min(indicies) if len(indicies) > 0 else 0

    def get_posts(self, i: int):
        """Get posts from Instagram page."""
        soup = BeautifulSoup(self.driver.page_source, "html.parser")
        for index, saved_post in self.posts.items():
            if index < i:
                continue
            self.code = saved_post["shortcode"]
            post = soup.find("a", role="link", href=self.post_link)
            if isinstance(post, NavigableString) or post is None:
                continue
            saved_post["post"] = str(post)
            saved_post["href"] = str(post["href"])
            img = post.find("img")
            if img is None or isinstance(img, NavigableString):
                continue
            saved_post["src"] = str(img["src"])
            if "alt" in img.attrs.keys():
                saved_post["alt"] = str(img["alt"])
            else:
                self.posts[index]["alt"] = ""
            self.post = saved_post

    def load_more(self):
        """Load more posts."""
        assert self.post is not None
        assert "href" in self.post.keys()
        assert self.post["href"] is not None
        last_post = self.driver.find_element(
            By.XPATH,
            f"//a[@href='{self.post['href']}']"
        )
        ActionChains(self.driver).scroll_to_element(last_post).perform()
        time.sleep(1)

    def intercept_posts(self) -> int:
        """Intercept requests to get data."""
        indicies = []
        for request in self.driver.requests:
            if ('api/v1/feed/user' in request.url
                    and request.response
                    and request.url not in self.requests):
                self.requests.append(request.url)
                body = decode(
                    request.response.body,
                    request.response.headers.get(
                        'Content-Encoding',
                        'identity'
                    )
                )
                data = json.loads(body)
                items = data.get("items")
                for item in items:
                    if not item.get('caption').get("text") in self.articles:
                        index = len(list(self.posts.keys()))
                        if index not in self.posts.keys():
                            self.posts[index] = {}
                        self.posts[index]["item"] = item
                        dt = datetime.fromtimestamp(
                            item.get(
                                'caption'
                            ).get(
                                'created_at_utc'
                            ),
                            tz=ZoneInfo("America/Los_Angeles")
                        )
                        self.dates.append(dt)
                        self.posts[index]["date"] = \
                            f"{dt.year:04d}-{dt.month:02d}-{dt.day:02d}"
                        if dt.hour > 12:
                            if dt.hour == 12:
                                self.posts[index]["time"] = \
                                    f"{dt.hour:02d}:{dt.minute:02d} PM"
                            else:
                                self.posts[index]["time"] = \
                                    f"{(dt.hour - 12):02d}:{dt.minute:02d} PM"
                        elif dt.hour == 0:
                            self.posts[index]["time"] = \
                                f"{12:02d}:{dt.minute:02d} AM"
                        else:
                            self.posts[index]["time"] = \
                                f"{dt.hour:02d}:{dt.minute:02d} AM"
                        self.posts[index]["article"] = \
                            item.get('caption').get("text")
                        self.articles.append(item.get('caption').get("text"))
                        self.posts[index]["comments"] = \
                            item.get("comment_count")
                        self.posts[index]["likes"] = \
                            item.get("like_count")
                        code = item.get('code')
                        self.posts[index]["shortcode"] = code
                        match item.get('media_type'):
                            case 2:
                                self.posts[index]["content"] = \
                                    f"https://www.instagram.com/reel/{code}"
                            case 8:
                                self.posts[index]["content"] = \
                                    f"https://www.instagram.com/p/{code}"
                            case 1:
                                self.posts[index]["content"] = \
                                    f"https://www.instagram.com/p/{code}"
                            case _:
                                self.posts[index]["content"] = ""
                        if (self.progress is not None
                                and self.task is not None
                                and isinstance(self.progress, Progress)):
                            if (hasattr(self, "date")
                                    and self.date is not None
                                    and hasattr(self, "days")
                                    and self.days is not None
                                    and isinstance(self.date, datetime)
                                    and isinstance(self.days, int)):
                                delta = datetime.fromisoformat(
                                    [
                                        post["date"]
                                        for post in
                                        self.posts.values()
                                        if "date" in post.keys()
                                    ][-1]
                                ) - self.date
                                self.progress.update(
                                    self.task,
                                    completed=(self.days - delta.days)
                                )
                            else:
                                self.progress.update(
                                    self.task,
                                    advance=1
                                )
        return min(indicies) if len(indicies) > 0 else 0

    def get_contents(self):
        """Get contents from Instagram page."""
        with Progress() as progress:
            self.progress = progress
            self.task = progress.add_task(
                "[yellow]Getting contents...[/yellow]",
                total=len(self.hrefs)
            )
            if self.cookies is None:
                self.cookies: Cookies = self.get_cookies()
            cookies = cookiejar_from_dict(self.cookies)
            cookies: Cookies = Cookies(cookies)
            with ThreadPoolExecutor(max_workers=5) as executor:
                futures = []
                for i, href in [
                    (key, post["href"])
                    for key, post in
                    self.posts.items()
                ]:
                    future = executor.submit(
                        get_post_data,
                        i,
                        self.path,
                        href,
                        cookies,
                        None,
                    )
                    futures.append(future)
                for future in futures:
                    future.add_done_callback(self.once_complete)
                d, nd = wait(futures, return_when="ALL_COMPLETED")
                executor.shutdown()
            self.progress.update(self.task, completed=(len(d) + len(nd)))
        self.save_step_two()

    def once_complete(self, f):
        """Update on completion."""
        try:
            i, p, t, dr = f.result()
            self.posts[i]["paths"] = p
            self.posts[i]["type_"] = t
            if dr is not None:
                self.posts[i]["download_request"] = dr
            self.progress.update(self.task, advance=1)
        except Exception:
            console.print(f.exception())
        finally:
            return None

    @property
    def download_requests(self) -> list[DownloadRequest]:
        """Get download requests."""
        return [
            post["download_request"]
            for post in self.posts.values()
            if "download_request" in post.keys()
        ]

    def download(self):
        """Create new download thread."""
        with Progress() as progress:
            drs = self.download_requests
            self.progress = progress
            self.task = progress.add_task(
                "Downloading posts...",
                total=len(drs)
            )
            if self.cookies is None:
                self.cookies: Cookies = self.get_cookies()
            cookies = cookiejar_from_dict(self.cookies)
            cookies: Cookies = Cookies(cookies)
            with ThreadPoolExecutor(max_workers=5) as executor:
                futures = []
                for download in drs:
                    for post in download.post:
                        future = executor.submit(
                            download_post_content,
                            post.url,
                            f"{download.path}/{post.file}",
                            cookies,
                        )
                        futures.append(future)
                for future in futures:
                    future.add_done_callback(self.once_downloaded)
                d, nd = wait(futures, return_when="ALL_COMPLETED")
                executor.shutdown()
            self.progress.update(self.task, completed=(len(d) + len(nd)))
        self.save_step_three()

    def once_downloaded(self, f):
        """Update when downloads complete."""
        try:
            if f.done():
                self.progress.update(self.task, advance=1)
        except Exception:
            console.print(f.exception())
        finally:
            return None

    def more(self):
        """Load more posts."""
        self.load_more()
        start = self.intercept_posts()
        self.get_posts(start)

    def start(self):
        """Start the scraper."""
        self.open()
        self.login()
        start = self.get_login_data()
        self.get_posts(start)

    def run(self):
        """Run the scraper."""
        self.start()
        with Progress() as progress:
            self.progress = progress
            self.task = self.progress.add_task(
                "Loading posts...",
                total=1
            )
            self.more()
            self.progress.update(self.task, completed=1)
            del self.progress
            del self.task
        self.save_step_one()
        self.close()

    def run_until(self, date: datetime):
        """Run until a certain date."""
        self.start()
        self.date = date
        delta = datetime.now() - self.date
        self.days: int = delta.days
        with Progress() as progress:
            self.progress = progress
            self.task = self.progress.add_task(
                "Loading posts...",
                total=self.days
            )
            while self.dates[-1] >= date:
                self.more()
            self.progress.update(self.task, completed=self.days)
            del self.progress
            del self.task
        self.save_step_one()
        self.close()

    def analyze(self):
        """Run step two."""
        self.get_contents()

    def post_link(self, href: str) -> bool:
        """Check if a link is a post."""
        return re.compile(
            r"(^\/p\/)|(^\/reel\/)"
        ).search(href) is not None and \
            href not in self.hrefs and \
            re.compile(
                f"({self.code})"
        ).search(href) is not None

    def generate(self, multi: bool = False) -> pd.DataFrame:
        """Generate a pandas dataframe from the scraped data."""
        index: List[int] = []
        columns: List[str] | pd.MultiIndex = [
            "href",
            "alt",
            "src",
            "article",
            "content",
            "date",
            "time",
            "item",
            "comments",
            "likes",
            "post",
            "paths",
            "type_"
        ]
        for i in self.posts.keys():
            index.append(i)
        data: List[List[Any]] = []
        for i in index:
            row: List[Any] = []
            for col in columns:
                if self.posts[i][col] is None:
                    row.append(None)
                else:
                    row.append(self.posts[i][col])
            data.append(row)
        if multi:
            columns = pd.MultiIndex.from_product([["scraped"], columns])
        return pd.DataFrame(data, index=index, columns=columns)

    def get_cookies(self) -> Cookies:
        """Get cookies from Selenium and add to requests."""
        cookies = {}
        selenium_cookies = self.driver.get_cookies()
        for cookie in selenium_cookies:
            cookies[cookie['name']] = cookie['value']
        return Cookies(cookies)

    @property
    def step_one(self) -> InstaStepOne:
        self.cookies: Cookies = self.get_cookies()
        return InstaStepOne(
            profile_page=self.profile_page,
            username=self.username,
            password=self.password,
            cookies=self.cookies,
            path=self.path,
            hrefs=self.hrefs,
            articles=self.articles,
            dates=self.dates,
            requests=self.requests,
            posts=self.posts
        )

    @property
    def step_two(self) -> InstaStepTwo:
        if self.cookies is None:
            self.cookies: Cookies = self.get_cookies()
        return InstaStepTwo(
            cookies=self.cookies,
            path=self.path,
            hrefs=self.hrefs,
            posts=self.posts,
        )

    @property
    def step_three(self) -> InstaStepThree:
        return InstaStepThree(
            hrefs=self.hrefs,
            articles=self.articles,
            dates=self.dates,
            posts=self.posts,
        )

    @property
    def step_four(self) -> InstaStepFour:
        posts = []
        with Progress(console=console) as progress:
            total = (len(self.posts) * 2) + 1
            load = progress.add_task("Converting data...", total=total)
            for i, post in self.posts.items():
                posts.append(
                    InstaPost(
                        index=i,
                        href=post["href"],
                        alt=post["alt"],
                        src=post["src"],
                        article=post["article"],
                        content=post["content"],
                        post=post["post"],
                        date=post["date"],
                        time=post["time"],
                        item=post["item"],
                        comments=post["comments"],
                        likes=post["likes"],
                        path=[Path(path) for path in post["paths"]],
                        type_=post["type_"]
                    )
                )
                progress.update(load, advance=1)

            four = InstaStepFour(
                url=self.profile_page,
                path=self.path,
                username=self.username,
                password=self.password,
                posts=posts,
                progress=progress,
                task=load
            )
            progress.update(load, completed=total)
        return four

    def save_step_one(self):
        """Save the first step."""
        sys.setrecursionlimit(100_000)
        obj = self.step_one
        with open(
            os.path.expanduser(f"{self.path}/step_one.pkl"),
            "wb"
        ) as f:
            pickle.dump(obj, f, protocol=pickle.DEFAULT_PROTOCOL)

    def step_one_saved(self) -> bool:
        """Check if step one is saved."""
        return os.path.isfile(os.path.expanduser(f"{self.path}/step_one.pkl"))

    def load_step_one(self):
        """Load the first step."""
        sys.setrecursionlimit(100_000)
        with read(
            os.path.expanduser(f"{self.path}/step_one.pkl"),
            "rb",
            description="Reading step one..."
        ) as f:
            obj = pickle.load(f)
        self.profile_page = obj.profile_page
        self.username = obj.username
        self.password = obj.password
        self.path = obj.path
        self.hrefs = obj.hrefs
        self.articles = obj.articles
        self.dates = obj.dates
        self.requests = obj.requests
        self.posts = obj.posts
        self.cookies: Cookies = Cookies(obj.cookies)

    def save_step_two(self):
        """Save the second step."""
        sys.setrecursionlimit(100_000)
        obj = self.step_two
        with open(
            os.path.expanduser(f"{self.path}/step_two.pkl"),
            "wb"
        ) as f:
            pickle.dump(obj, f, protocol=pickle.DEFAULT_PROTOCOL)

    def step_two_saved(self) -> bool:
        """Check if step two is saved."""
        return os.path.isfile(os.path.expanduser(f"{self.path}/step_two.pkl"))

    def load_step_two(self):
        """Load the second step."""
        sys.setrecursionlimit(100_000)
        with read(
            os.path.expanduser(f"{self.path}/step_two.pkl"),
            "rb",
            description="Reading step two..."
        ) as f:
            obj = pickle.load(f)
        self.cookies = obj.cookies
        self.path = obj.path
        self.hrefs = obj.hrefs
        self.posts = obj.posts

    def save_step_three(self):
        """Save the third step."""
        sys.setrecursionlimit(100_000)
        obj = self.step_three
        with open(
            os.path.expanduser(f"{self.path}/step_three.pkl"),
            "wb"
        ) as f:
            pickle.dump(obj, f, protocol=pickle.DEFAULT_PROTOCOL)

    def step_three_saved(self) -> bool:
        """Check if step three is saved."""
        return os.path.isfile(
            os.path.expanduser(
                f"{self.path}/step_three.pkl"
            )
        )

    def load_step_three(self):
        """Load the third step."""
        sys.setrecursionlimit(100_000)
        with read(
            os.path.expanduser(f"{self.path}/step_three.pkl"),
            "rb",
            description="Reading step three..."
        ) as f:
            obj = pickle.load(f)
        self.hrefs = obj.hrefs
        self.articles = obj.articles
        self.dates = obj.dates
        self.posts = obj.posts

    def save_step_four(self):
        """Save the fourth step."""
        sys.setrecursionlimit(100_000)
        obj = self.step_four
        with open(
            os.path.expanduser(f"{self.path}/step_four.pkl"),
            "wb"
        ) as f:
            pickle.dump(obj, f, protocol=pickle.DEFAULT_PROTOCOL)

    def step_four_saved(self) -> bool:
        """Check if step four is saved."""
        return os.path.isfile(
            os.path.expanduser(
                f"{self.path}/step_four.pkl"
            )
        )

    def load_step_four(self):
        """Load the fourth step."""
        sys.setrecursionlimit(100_000)
        with read(
            os.path.expanduser(f"{self.path}/step_four.pkl"),
            "rb",
            description="Reading step four..."
        ) as f:
            obj = pickle.load(f)
        with Progress(console=console) as progress:
            total = len(obj.posts)
            serialize = progress.add_task(
                "Deserializing step four...",
                total=total
            )
            hrefs = []
            articles = []
            dates = []
            for post in obj.posts.values():
                self.posts[post.index] = {
                    "href": post.href,
                    "alt": post.alt,
                    "src": post.src,
                    "article": post.article,
                    "content": post.content,
                    "date": post.date,
                    "time": post.time,
                    "item": post.item,
                    "comments": post.comments,
                    "likes": post.likes,
                    "post": post.post,
                    "paths": post.path,
                    "type_": post.type_,
                }
                hrefs.append(post.href)
                articles.append(post.article)
                dates.append(datetime.strptime(
                    f"{post.date} {post.time}",
                    "%Y-%m-%d %I:%M %p"
                ))
                progress.update(serialize, advance=1)
            self.profile_page = obj.url
            self.path = obj.path
            self.username = obj.username
            self.password = obj.password
            self.hrefs = hrefs
            self.articles = articles
            self.dates = dates
            progress.update(serialize, completed=total)


def download_post_content(
    link: str,
    file: str,
    cookies: Cookies,
):
    """Download posts."""
    r = requests.get(
        link,
        allow_redirects=True,
        stream=True,
        cookies=cookies
    )
    if r.status_code >= 400:
        cookies.refresh(link)
    dir = file.rsplit("/", 1)[0]
    if not os.path.exists(dir):
        os.makedirs(dir)
    with open(file, 'wb') as f:
        for chunk in r.iter_content(1024):
            if chunk:
                f.write(chunk)


PostData = Tuple[
    int,
    List[PathLike | str],
    Literal["image", "video", ""],
    DownloadRequest | None
]


def get_post_data(
    num: int,
    path: PathLike,
    href: str,
    cookies: Cookies,
    wait: int | None,
) -> PostData:
    """Get data from post."""
    save_path = os.path.expanduser(f"{path}/{num}")
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    if len(os.listdir(save_path)) == 0:
        return make_request(
            num,
            path,
            href,
            cookies,
            wait,
        )
    else:
        return add_paths(
            num,
            path,
            href,
            cookies,
            wait,
        )


def make_request(
    num: int,
    path: PathLike,
    href: str,
    cookies: Cookies,
    wait: int | None,
) -> PostData:
    """Get post contents."""
    if wait is not None:
        time.sleep(wait)
    link = f"https://instagram.com{href}"
    res = requests.get(
        link,
        allow_redirects=True,
        cookies=cookies,
    )
    content = res.content
    soup = BeautifulSoup(content, "html5lib")
    soup = soup.find("script", {"type": "application/ld+json"})
    if soup is None:
        if res.status_code == 200:
            wait = diagnose_wait(wait, res.status_code, cookies, link)
            return make_driver(
                num,
                path,
                href,
                cookies,
                wait
            )
        else:
            wait = diagnose_wait(wait, res.status_code, cookies, link)
            return make_request(
                num,
                path,
                href,
                cookies,
                wait,
            )
    elif isinstance(soup, NavigableString):
        soup = str(soup)
    else:
        soup = str(soup.string)
    soup = json.loads(soup)
    soup = soup[0]
    video = soup.get("video")
    image = soup.get("image")
    if video is not None and len(video) != 0:
        return get_video(num, video[0], path)
    elif image is not None and len(image) != 0:
        if len(image) > 1:
            return get_carousel(num, image, path)
        else:
            return get_photo(num, image[0], path)
    else:
        return ask_user_for_post(num, href, soup, path, cookies)


def make_driver(
    num: int,
    path: PathLike,
    href: str,
    cookies: Cookies,
    wait: int | None,
) -> PostData:
    """Get post contents."""
    if wait is not None:
        time.sleep(wait)
    options = Options()
    options.page_load_strategy = 'eager'
    driver = webdriver.Chrome(options=options)
    link = f"https://instagram.com{href}"
    driver.get(link)
    WebDriverWait(driver, 10).until(
        EC.presence_of_element_located(
            (
                By.XPATH,
                "//main[contains(@role, 'main')]"
            )
        )
    )
    soup = BeautifulSoup(driver.page_source, "html5lib")
    if soup.find("div", {"class": "_aap0"}) is not None:
        return get_carousel_from_page(soup, num, path)
    elif soup.find("div", {"class": "_aatk"}) is not None:
        return get_video_from_page(soup, num, path)
    else:
        wait = diagnose_wait(wait, 200, cookies, link)
        return make_request(
            num,
            path,
            href,
            cookies,
            wait,
        )


def get_video_from_page(
    soup,
    num: int,
    path: PathLike,
) -> PostData:
    """Get video from page."""
    soup = soup.find("div", {"class": "_aatk"})
    if soup is None:
        raise Exception("No video found.")
    elif isinstance(soup, NavigableString):
        video = str(soup)
    else:
        soup = soup.find("video")
        if soup is None:
            raise Exception("No video found.")
        elif isinstance(soup, NavigableString):
            video = str(soup)
        else:
            video = str(soup["src"])
    try_file = re.compile(
        r"(?=\/([0-9A-Za-z-_]*\.mp4))"
    ).search(video)
    if try_file is None:
        file = Path("content.mp4")
    else:
        file = Path(try_file.group(1))
    path = Path(os.path.expanduser(f'{path}/{num}'))
    paths: List[PathLike | str] = [f"{path}/{file}"]
    types = "video"
    download_requests = DownloadRequest(
        path=path,
        post=[Post(
            url=video,
            file=file
        )]
    )
    return num, paths, types, download_requests


def get_carousel_from_page(
    soup,
    num: int,
    path: PathLike,
) -> PostData:
    """Get carousel from page."""
    soup = soup.find("ul", {"class": "_acay"})
    if soup is None or isinstance(soup, NavigableString):
        raise Exception("No carousel found.")
    else:
        soup = soup.find_all("img")
        if soup is None or isinstance(soup, NavigableString):
            raise Exception("No carousel found.")
        else:
            posts = soup
    paths: List[PathLike | str] = []
    path = Path(os.path.expanduser(f'{path}/{num}'))
    download_requests = DownloadRequest(
        path=path,
        post=[]
    )

    for i, post in enumerate(posts):
        url = post.get("src")
        try_file = re.compile(
            r"(?=\/([0-9A-Za-z-_]*\.jpg))"
        ).search(url)
        if try_file is None:
            file = Path(f"content{i}.jpg")
        else:
            file = Path(try_file.group(1))
        paths.append(f"{path}/{file}")
        download_requests.post.append(Post(
            url=url,
            file=file
        ))
    types = "image"
    return num, paths, types, download_requests


def diagnose_wait(
    wait: int | None,
    status: int,
    cookies: Cookies,
    link: str | None = None
) -> int:
    """Diagnose how long to wait."""
    if wait is None:
        wait = 2
    else:
        wait = wait ** 2
    if status == 429:
        if cookies.refresh_count() < 10:
            if link is not None:
                cookies.refresh(link)
            else:
                cookies.refresh()
    if wait > 7200:
        raise Exception(
            """
Wait is over two hours.
Please wait a few hours until continuing the program
            """)
    return wait


def ask_user_for_post(
    num: int,
    href: str,
    soup: Dict,
    path: PathLike,
    cookies: Cookies,
) -> PostData:
    """Ask user about the post."""
    t = console.input(Prompt.ask(
        f"""Unhandled post type: {href}
        Is this a reel/video, a carousel, or normal image post?
        (press r for reel or video, c for carousel, i for image)

        Go ahead and copy and paste the link to see the post.

        A carousel is a post with multiple images""",
        choices=["r", "reel", "video", "c", "carousel", "i", "image"],
        show_choices=True
    ))
    if t in ["r", "reel", "video"]:
        video = soup.get("video")
        if video is None:
            raise Exception("Unable to fetch post")
        return get_video(num, video[0], path)
    elif t in ["c", "carousel"]:
        images = soup.get("image")
        if images is None:
            raise Exception("Unable to fetch post")
        return get_carousel(num, images, path)
    elif t in ["i", "image"]:
        images = soup.get("image")
        if images is None:
            raise Exception("Unable to fetch post")
        return get_photo(num, images[0], path)
    else:
        yn = console.input(Prompt.ask(
            "Would you like to stop the program?",
            choices=[
                "y",
                "Y",
                "yes",
                "Yes",
                "YES",
                "n",
                "N",
                "no",
                "No",
                "NO"
            ],
            show_choices=True
        ))
        if yn in ["y", "Y", "yes", "Yes", "YES"]:
            raise Exception("Stopping")
        elif yn in ["n", "N", "no", "No", "NO"]:
            return make_request(num, path, href, cookies, 2)
        else:
            return ask_user_for_post(
                num,
                href,
                soup,
                path,
                cookies,
            )


def add_paths(
    num: int,
    path: PathLike,
    href: str,
    cookies: Cookies,
    wait: int | None,
) -> PostData:
    """Add paths to List."""
    paths: List[PathLike | str] = []
    save_path = os.path.join(path, str(num))
    for root, dirs, files in os.walk(
        save_path,
        topdown=False
    ):
        for name in files:
            paths.append(os.path.join(root, name))
        for name in dirs:
            paths.append(os.path.join(root, name))
    extension = str(paths[0]).rsplit(".", 1)[1]
    if extension == "jpg":
        rt = "image"
    elif extension == "mp4":
        rt = "video"
    else:
        t = console.input(Prompt.ask(
            f"""What is the file type?
            [bold]{extension}[/bold]""",
            choices=["image", "video", "i", "v", "again", "a"]
        ))
        if t in ["i", "image"]:
            rt = "image"
        elif t in ["v", "video"]:
            rt = "video"
        elif t in ["again", "a"]:
            return make_request(
                num,
                path,
                href,
                cookies,
                wait,
            )
        else:
            return add_paths(
                num,
                path,
                href,
                cookies,
                wait
            )
    return num, paths, rt, None


def get_video(
        num: int,
        video: Dict,
        path: PathLike
) -> PostData:
    """Get video from Instagram post."""
    url = str(video.get("contentUrl"))
    try_file = re.compile(
        r"(?=\/([0-9A-Za-z-_]*\.mp4))"
    ).search(url)
    thumburl = str(video.get("thumbnailUrl"))
    try_thumb_file = re.compile(
        r"(?=\/([0-9A-Za-z-_]*\.jpg))"
    ).search(url)
    if try_file is None:
        file = Path("content.mp4")
    else:
        file = Path(try_file.group(1))
    if try_thumb_file is None:
        thumbfile = Path("thumbnail.jpg")
    else:
        thumbfile = Path(try_thumb_file.group(1))
    path = Path(os.path.expanduser(f'{path}/{num}'))
    paths: List[PathLike | str] = [f"{path}/{file}"]
    types = "video"
    download_requests = DownloadRequest(
        path=path,
        post=[Post(
            url=url,
            file=file
        ), Post(
            url=thumburl,
            file=thumbfile
        )]
    )
    return num, paths, types, download_requests


def get_photo(
        num: int,
        image: Dict,
        path: PathLike
) -> PostData:
    """Download photo from post."""
    url = str(image.get("url"))
    try_file = re.compile(r"(?=\/([0-9A-Za-z-_]*\.jpg))").search(url)
    if try_file is None:
        file = Path("content.jpg")
    else:
        file = Path(try_file.group(1))
    path = Path(os.path.expanduser(f'{path}/{num}'))
    paths: List[PathLike | str] = [f"{path}/{file}"]
    types = "image"
    download_requests = DownloadRequest(
        path=path,
        post=[Post(
            url=url,
            file=file
        )]
    )
    return num, paths, types, download_requests


def get_carousel(
        num: int,
        images: List[Dict],
        path: PathLike
) -> PostData:
    """Get all images in a carousel post."""
    path = Path(os.path.expanduser(f'{path}/{num}'))
    paths = []
    down_req = DownloadRequest(
        path=path,
        post=[]
    )
    for i, el in enumerate(images):
        url = str(el.get("url"))
        try_file = re.compile(
            r"(?=\/([0-9A-Za-z-_]*\.jpg))"
        ).search(url)
        if not try_file:
            file = Path(f"content{i}.jpg")
        else:
            file = Path(try_file.group(1))
        down_req.post.append(Post(
            url=url,
            file=file
        ))
        paths.append(f"{path}/{file}")
    types = "image"
    return num, paths, types, down_req
