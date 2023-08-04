# InstaCrawl

An [Instagram] post crawler focused on optimizing algorithm performance through model classification

## Installation

**With pip**
```bash
pip install instacrawl
```

**From source**
```bash
git clone https://github.com/annie444/instacrawl.git 

cd instacrawl

pip install -r requirements.txt

python instacrawl/__main__.py
```

**NOTE:** InstaCrawl requires [Google Chrome] and [chromedriver] to be installed. The [PyPi] package comes with an installation of [chromedriver], however, there is no guarantee that this version of [chromedriver] will be compatible with your version of [Google Chrome].

## Using InstaCrawl

### Setup

To launch InstaCrawl, open a new terminal window and type `instacrawl`.

You'll end up at a selection dialogue to select where you'd like to start. The reason for this is that InstaCrawl can take a while to get through all the steps in some cases. This way you can stop after each step and resume the analysis later.

```bash
╭───────────────────────────────── InstaCrawl ─────────────────────────────────╮
│ What would you like to do?                                                   │
│                                                                              │
│                                                                              │
│                                                                              │
│                                                                              │
│                                                                              │
│                                                                              │
│                                                                              │
│                                                                              │
│                                                                              │
│ Press 1 to start crawling a new profile                                      │
│ Press 2 to scrape saved data from a crawl                                    │
│ Press 3 to download posts from a crawl                                       │
│ Press 4 to align a crawl                                                     │
│ Press 5 to analyze a crawl                                                   │
│ Press 6 to view the results from a crawl                                     │
│ Press 0 to exit                                                              │
│                                                                              │
╰──────────────────────────────────────────────────────────────────────────────╯
```

### Step 1.

After selecting step one you'll be prompted to enter the account you wish to analyze alongside the credentials to the instagram account you'd like to use for accessing the account. Your login info will be saved in a *~/.instacrawlrc* file for future reference. You then will be prompted for a directory to save the data to as well as a date you'd like to crawl back to. You can leave the date empty to just crawl the first ~21 posts on the account.

During step 1 [Google Chrome] will open repeatedly and it will say that the site is unsafe. (the HTTPS/security chip on the URL bar will be red.) This is expected. At this point, InstaCrawl is using a proxy to intercept all data between your computer and [Instagram] to parse out your posts. Facebook uses a clever technique of changing the way they send posts from their servers to your computer. To get around this, InstaCrawl may either: 
  1. take longer than usual to parse out a post, which very well may look like it's stalled. Do not worry, InstaCrawl tries many different parsing methods to sort through the garbage and find the necessary information.
  2. [Google Chrome] may open and close a few times. This is usually the result of [Instagram] catching onto the crawler and trying to block InstaCrawl's attempts to scrape data. In order to circumvent this, InstaCrawl will do multiple rounds of deleting cookies and changing proxies.


At the end of step 1, InstaCrawl will save all of the scraped data in a `step_one.pkl` file. There are programs that can open these types of files, but all that's been collected thus far is:
  - The post codes
  - The number of likes
  - The number of comments
  - The basic post metadata

### Step 2.

The point of point two is to scrape the actual media from the posts. Or at least to prepare to do that. This step in particular often upsets the servers, so during this step InstaCrawl will most likely repeatedly refresh it's cookies. InstaCrawl will try to do as much of step 2 in the background, but it's likely that [Google Chrome] will open at least once during this step.

After step 2, InstaCrawl will save the collected post data in a `step_two.pkl` file. At this point all of the publicly available data has been collected. In addition to step 1, step 2 has collected:
  - The hashed download links to post media (i.e. images and videos)
  - The alt text and accessibility information tied to each post
  - The page HTML and server JSON for that post
  - The type of post (e.g. carousel, reel, etc.)

### Step 3.

From here on out, InstaCrawl will mostly work in the background. During step 3 InstaCrawl will open as many threads as possible to download post content in parallel. You'll notice at this point that the folder with all your save data will have numbered folders containing the high quality content from each post in either `.jpg` or `.mp4` format. Each folder number lines up with the indices in the data.

### Step 4.

During step 4 you'll get to look at the posts in the terminal (it's not the best, but come on, I got an image to print out on any tty output.) To be honest, this is mostly a gimmick and you can press `SPACE` to continue. Next you'll be prompted for a save path of the data thus far. This can be in any format supported by [Pandas]. At this point you can stop if you just wanted the data from the posts, but if you are here for algorithm analytics, keep moving forward.

### Step 5.

Step 5 is the most computationally intense and it may make your computer very slow while it's running. Because of this, I advise that you use something like [this app] to keep your computer awake, or at the very least, just plug in your computer and set it to not fall asleep. 

At the beginning of step 5 you'll probably get a lot of weird errors and warnings. Don't worry, nothing is broken (I hope), InstaCrawl is just initializing [dlib] and [huggingface] for analysing your posts against facebook's post analysis models. This part generally looks like it's doing a lot, but it really is just downloading models and packages from the web. You may also notice your RAM being eaten up, but generally little CPU usage. This is InstaCrawl preprocessing all of the posts. Don't worry though, everything will hop out of RAM and into some tempfiles soon.

After everything has downloaded and initialized, InstaCrawl will begin running your posts through a slew of machine learning models. Each model's output will be stored in a human-readable format of your choosing (much like it step 4), but there will inevitably be **A LOT** of data to store. So be prepared, the final output of a business profile's analysis is usually around 3GB.

Once complete you'll be prompted for a path and file to save the end data in. This, again, can be in any format supported by [Pandas] and will contain all of the data from step. At this point, this is the completed data.

### Step 6.

Step 6 does nothing itself other than launch an instance of [D-tale] with all of the fancy add-ons for you to explore, analyse, and visualize the data. Feel free to come back to step 6 as often as you'd like. Also, note that InstaCrawl comes with all of the necessary packages to export graphs, run statistical analyses, and parse/sort the data. No need to install any additional software/packages.


## Reference

If you want to build of InstaCrawl, you're welcome to! I built it using common OO schemes (although OO isn't my favorite). However, I will warn you that isn't a bit of a nightmare in terms of complex datatypes. But for a weekend project, it is what it is. 

InstaCrawl is built upon three main classes. The first being the base `InstaCrawl` class in the `prompts.py` file. This class simply handles the control flow of the application and generally isn't that interesting in terms of development, so I've chosen to omit it's docs. If you really want the docs, feel free to open an issue.

### /insta.py

```python

class Cookies(Dict):
    """
    Class to store cookies.

    This class inherits the Dict class from the typing module,
    and provides a few helpful abstractions for refreshing the
    cookies and handling parallel threads attached to a shared
    instance. Convenient when maintaining a singleton pattern.

    ...
    Attributes
    ----------
    cookies : dict[str, Optional[str]]
      This is a simple dictionary of cookies with the
      `cookie_name`: `cookie_value` pattern

    count : int, optional, default: 0
      This is used to maintain a count of how often the 
      cookies have been refreshed. This is useful if you're 
      trying to limit the amount of calls to instagram.com

    Methods
    -------
    refresh(
      link : str, optional, default = "https://instagram.com"
    ) -> None

    refresh_count() -> int
    """

    def refresh(self, link: str = "https://instagram.com"):
        """
        A method to refresh the cookies with Selenium.
        ...
        Args
        ----
          link : str, optional, default = "https://instagram.com"

        Returns
        -------
          None

        """

    def refresh_count(self):
        """
        Return number of times cookies have been refreshed.
        ...
        Returns
        -------
          int : the number of times the cookies have been refreshed
        """


class InstagramData:
    """
    Class to store and scrape data from Instagram profile page.

    This class does all the heavy lifting when it comes to web scraping.
    ...
    Attributes
    ----------
      save_to : PathLike
        This is the path where the saved data is stored.

    Properties
    ----------
      step_one : InstaStepOne
      step_two : InstaStepTwo
      step_three : InstaStepThree
      step_four : InstaStepFour
      download_requests : List[DownloadRequests]

    Methods
    -------
      new(
        profile_url : str,
        save_path : Optional[PathLike],
        username : str,
        password : str
      ) -> None
      open() -> None
      close() -> None
      login() -> None
      get_login_data() -> int
      get_posts( i : int ) -> None
      load_more() -> None
      intercept_posts() -> int
      get_contents() -> None
      once_complete(
        f : concurrent.futures.Future
      ) -> None
      download() -> None
      once_downloaded(
        f : concurrent.futures.Future
      ) -> None
      more() -> None
      start() -> None
      run() -> None
      run_until( date : datetime.datetime ) -> None
      analyze() -> None
      post_link( href : str ) -> bool
      generate( multi : bool ) -> pandas.DataFrame
      get_cookies() -> Cookies
      save_step_one() -> None
      step_one_saved() -> bool
      load_step_one() -> None
      ... same methods exist for step_two through step_four

    """

    def open(self) -> None:
      """Opens Selenium, ChromeDriver, and a requests session"""

    def close(self) -> None:
      """
      Closes the Selenium and requests sessions 
      and wipes them from memory
      """

    def login(self) -> None:
      """Runs the login workflow with Selenium"""

    def get_login_data(self) -> int:
      """
      Intercepts the login data from the server.
      ...
      Returns
      -------
        int
          The key from the self.posts dictionary for the first
          post processed.
      """

    def get_posts(self, i: int) -> None:
      """
      Scrape the HTML for post data starting at post `i'
      """

    def load_more(self) -> None:
      """Load more posts"""

    def intercept_posts(self) -> int:
      """
      Intercept posts as they are sent from the server.
      ...
      Returns
      -------
        int
          The key from the self.posts dictionary for the first
          post processed.
      """

    def get_contents(self) -> None:
      """
      Get the post contents from each individual posts' page.

      NOTE: Runs in parallel threads
      """

    def once_complete(self, f: concurrent.futures.Future) -> None:
      """Process the data returned from `get_contents()'"""

    def download(self) -> None:
      """
      Download each posts' media

      NOTE: Runs in parallel
      """

    def once_downloaded(self, f: concurrent.futures.Future) -> None:
      """Process the data returned from `download()'"""

    def more(self) -> None:
      """Compounded method to load more posts and scrape their data"""

    def start(self) -> None:
      """
      Compound method that:
        - Logs into Instagram
        - Collects the first round of data
      """

    def run(self) -> None:
      """Auto grabs the first ~21 posts"""

    def run_until( date : datetime.datetime ) -> None
      """
      Crawls posts until it hits a given date.

      Args
      ----
        data : datetime.datetime 
          When to crawl until (e.g. 2021-04-04)
      """

    def analyze(self) -> None:
      """This is just an alias for `get_contents()'"""

    def post_link(self, href: str) -> bool:
      """
      This is a utility method to filter the `href' 
      attributes of anchor tags with BeautifulSoup
      """

    def generate(self, multi: bool) -> pandas.DataFrame:
      """
      Generates a pandas.DataFrame from the data

      Args
      ----
        multi : bool
          Whether or not you would like the resulting DataFrame
          to have a multi-index on the columns

      Returns
      -------
        pd.DataFrame
      """

    def get_cookies(self) -> Cookies:
      """
      Initializes the Cookies class from the cookies stored
      in the selenium and requests sessions
      """

    def save_step_one(self) -> None:
      """Writes to step_one.pkl"""

    def step_one_saved(self) -> bool:
      """Alias for os.path.exists"""

    def load_step_one(self) -> None:
      """Loads all the saved data from the previous step"""
```

### /analysis.py

```python
class Postalyzer:
    """
    The analysis control class for running multithreaded inference

    Attributes
    ----------
      data : InstaStepFour
        The data from step 4
    
    Methods
    -------
      analyze() -> List[pandas.DataFrame]
    """

    def analyze(self) -> List[pd.DataFrame]:
      """
      Runs and controls the analysis of the data
      
      NOTE: Runs on multiple threads

      Returns
      -------
        List[pandas.DataFrame]
          A list of the results from each class of analysis
      """


class InstaAnalysis(object):
  """
  A base class for each subsequent analysis pipeline

  Attributes
  ----------
    data : InstaStepFour
      The data from step 4 
    name : str
      The name of the analysis target (e.g. 'images')

  Methods
  -------
    insert_result(
      post: int,
      cat: str,
      label: str,
      data: Any,
      append: bool = False
    ) -> None
    analyze() -> None
    save() -> pandas.DataFrame
  """

    def insert_result(
        self,
        post: int,
        cat: str,
        label: str,
        data: Any,
        append: bool = False
    ) -> None:
      """Helper method for managing results"""

    def analyze(self) -> None:
      """The actual ML algorithm"""

    def save(self) -> pd.DataFrame:
      """
      Saves the results to a pandas.DataFrame

      Returns
      -------
        pandas.DataFrame
          The results from the ML algorithm
      """

# Not shown here is three helper methods for displaying the progress
# to the terminal.

class VideoAnalysis(InstaAnalysis):
  """
  Runs analysis of videos utilizing the
    MCG-NJU/videomae-base-finetuned-kinetics
  model
  """


class ImageAnalysis(InstaAnalysis):
  """
  Runs analysis on images utilizing the following models:
    - facebook/detr-resnet-50
    - mattmdjaga/segformer_b2_clothes
    - face_recognition_models
  """


class TextAnalysis(InstaAnalysis):
  """
  Runs analysis on the post's text utilizing:
    - SamLowe/roberta-base-go_emotions
    - alimazhar-110/website_classification
    - cardiffnlp/twitter-roberta-base-sentiment-latest
  """
```

[Google Chrome]: https://www.google.com/chrome/
[chromedriver]: https://chromedriver.chromium.org
[Instagram]: https://instagram.com
[PyPi]: https://pypi.org/project/instacrawl/
[this app]: https://www.google.com/url?sa=t&rct=j&q=&esrc=s&source=web&cd=&cad=rja&uact=8&ved=2ahUKEwjr56Lfi8SAAxV5C0QIHdvcBG4QFnoECBgQAQ&url=https%3A%2F%2Fapps.apple.com%2Fus%2Fapp%2Famphetamine%2Fid937984704%3Fmt%3D12&usg=AOvVaw2o99yDNP2d-ILjXKc5IE0I&opi=89978449
[Pandas]: https://pandas.pydata.org/docs/user_guide/io.html
[huggingface]: https://huggingface.co
[D-tale]: https://github.com/man-group/dtale
[dlib]: http://dlib.net
