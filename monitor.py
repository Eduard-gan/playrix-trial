#!/usr/bin/python3

import argparse
import decimal
import logging
import os
import sys
from collections import defaultdict
from datetime import datetime, date
from time import sleep, time
from typing import Optional, List, Union, Dict, Tuple, Any, Type, Set
from urllib.parse import urlparse, urlencode

import requests
from requests.auth import HTTPBasicAuth


GITHUB_TIMESTAMP_FORMAT = "%Y-%m-%dT%H:%M:%SZ"


def parse_args() -> argparse.Namespace:
    """Принимает аргументы переданные из командной строки и возвращает их в объекте argparse.Namespace."""

    parser = argparse.ArgumentParser(
        prog="monitor.py",
        description="Просмотр статистики репозитория на Github.",
    )

    parser.add_argument(
        "url",
        help="URL публичного репозитория на github.com.",
    )
    parser.add_argument(
        "-s",
        "--start_date",
        help="Дата начала анализа. Если пустая, то неограничено. Пример: 15.01.2020",
        type=lambda x: datetime.strptime(x, '%d.%m.%Y').date(),
        default=datetime.fromtimestamp(0).date(),
    )
    parser.add_argument(
        "-e",
        "--end_date",
        help="Дата начала анализа. Если пустая, то неограничено. Пример: 15.01.2021",
        type=lambda x: datetime.strptime(x, '%d.%m.%Y').date(),
        default=datetime.now().date(),
    )
    parser.add_argument(
        "-b",
        "--branch",
        help="Ветка репозитория. По умолчанию - master.",
        default="master",
    )

    return parser.parse_args()


def github_api_request(path: str, method: str = "GET", params: Optional[Dict] = None) -> Union[Dict, List]:
    """
    Выполняет запрос к API GitHub.

    Если заданы переменные окружения GITHUB_USERNAME и GITHUB_PERSONAL_TOKEN - выполняются аутентифицированные
    запросы с увеличенными лимитами.

    Происходят ретраи для сетевых ошибок, а так же для ошибок превышения лимитов на запросы к API.

    :raises SearchResultsLimitReached в случае логической ошибки в
    работе с search API.(Попытка получения более 1000 результатов)
    """

    retry_time = 5  # Дефолтное время ожидания в случае сетевой ошибки или ошибки превышения лимитов на запросы.

    username = os.getenv("GITHUB_USERNAME")
    personal_token = os.getenv("GITHUB_PERSONAL_TOKEN")

    kwargs = dict(
        url=f"https://api.github.com{path}",
        params=urlencode(params, safe=":+") if params else None,
        method=method,
        timeout=(3.5, 10),
    )
    if username and personal_token:
        kwargs['auth'] = HTTPBasicAuth(username=username, password=personal_token)

    while True:
        try:
            response = requests.request(**kwargs)
            response.raise_for_status()
            logging.info(f"Осталось запросов для {path}: {response.headers['x-ratelimit-remaining']}")
            break

        except (requests.exceptions.ConnectionError, requests.exceptions.ReadTimeout) as e:
            logging.error(repr(e))
            logging.info(f"Повторный запрос через {retry_time} сек.")
            sleep(retry_time)

        except requests.HTTPError as e:
            if e.response.json()['message'].startswith("API rate limit exceeded"):
                reset_at = int(e.response.headers['x-ratelimit-reset'])
                current_time = int(time())
                wait_time = reset_at - current_time
                logging.error(f"Исчерпан лимит запросов к API Github. "
                              f"Восстановление лимита в {datetime.fromtimestamp(reset_at)}")
                if wait_time > 0:
                    logging.info(f"Повторный запрос через {wait_time} сек.")
                    sleep(wait_time)
                else:
                    logging.info(f"Повторный запрос через {retry_time} сек.")
                    sleep(retry_time)
            elif e.response.json()['message'].startswith("Only the first 1000 search results are available"):
                raise SearchResultsLimitReached from None
            else:
                raise

    return response.json()


class SearchResultsLimitReached(Exception):
    """
    Произошла попытка получения более 1000 результатов поискового запроса через пагинацию.

    Если в ответе на поисковой запрос было более 1000 результатов,
    то потребуется более одного запроса для их получения.
    """


class GithubObject:
    """Объект в API GitHub."""

    node_id: str
    created_at: Optional[datetime] = None

    api_path: Optional[str] = None

    def __init__(self, node_id: str, created_at: Optional[datetime] = None) -> None:
        self.node_id = node_id
        self.created_at = created_at

    def __hash__(self) -> int:
        return hash(self.node_id)

    def __eq__(self, other) -> bool:
        return True if self.node_id == other.node_id else False

    @classmethod
    def from_dict(cls, data: Dict, **kwargs) -> "GithubObject":
        """
        Возвращает Инстанс соответствующего дочернего класса.

        :param data: Словарь полученный от API и содержащий данные об этом объекте.
        """

        raise NotImplementedError


class Contributor(GithubObject):
    """Зарегистрированный пользователь Github.com, который является автором коммита в репозитории."""

    login: str
    contributions: int = 0

    def __init__(self, login: str, contributions: int = 0, **kwargs):
        super().__init__(**kwargs)
        self.login = login
        self.contributions = contributions

    def __repr__(self):
        return f"{self.__class__.__name__}(login={self.login}, contributions={self.contributions})"


class Commit(GithubObject):
    """
    Отдельный набор изменений в репозитории.

    Может быть сделан и не зарегистрированным на Github.com пользователем.
    В таком случае он условно считается анонимным т.к. найти логин такого пользователя нельзя.
    """

    contributor: Optional[Contributor] = None

    api_path = "commits"

    def __init__(self, contributor: Optional[Contributor] = None, **kwargs):
        super().__init__(**kwargs)
        self.contributor = contributor

    @classmethod
    def from_dict(cls, data: Dict, **kwargs) -> "Commit":
        return cls(
            node_id=data['node_id'],
            created_at=datetime.strptime(data['commit']['author']['date'], GITHUB_TIMESTAMP_FORMAT),
            contributor=Contributor(
                node_id=data['author']['node_id'],
                login=data['author']['login'],
                ) if data.get('author') else None,
        )


class Issue(GithubObject):
    """Задача в репозитории."""

    state: str
    closed_at: Optional[datetime] = None

    api_path: str = "issues"

    def __init__(self, state: str, closed_at: Optional[datetime] = None, **kwargs):
        super().__init__(**kwargs)
        self.state = state
        self.closed_at = closed_at

    @staticmethod
    def get_data(data: Dict) -> Dict:
        """Получает необходимые даные для создания инстанса из словаря, который приходит от API."""

        return dict(
            node_id=data['node_id'],
            created_at=datetime.strptime(data['created_at'], GITHUB_TIMESTAMP_FORMAT),
            closed_at=datetime.strptime(data['closed_at'], GITHUB_TIMESTAMP_FORMAT) if data.get('closed_at') else None,
            state=data['state'],
        )

    @classmethod
    def from_dict(cls, data: Dict, **kwargs) -> "Issue":
        return cls(**cls.get_data(data))

    @property
    def is_old(self) -> bool:
        """
        Определяет, считается ли задача 'старой'.

        Issue считается старым, если он не закрывается в течение 14 дней.
        """

        overdue_days = 14

        if self.closed_at:
            if (self.closed_at - self.created_at).days >= overdue_days:
                return True
        else:
            if (datetime.now() - self.created_at).days >= overdue_days:
                return True

        return False


class PullRequest(Issue):
    """
    Pull-request - это разновидность задачи.
    Дополнительно несет информацию о своей базовой ветке.
    """

    base: str

    def __init__(self, base: str, **kwargs):
        super().__init__(**kwargs)
        self.base = base

    @classmethod
    def from_dict(cls, data: Dict, **kwargs) -> "PullRequest":
        return cls(**cls.get_data(data=data), base=kwargs["base"])

    @property
    def is_old(self) -> bool:
        """
        Определяет, считается ли пулл-реквест 'старым'.

        Pull request считается старым, если он не закрывается в течение 30 дней и до сих пор открыт.
        """

        if (datetime.now() - self.created_at).days >= 30 and self.state == "open":
            return True
        else:
            return False


class GithubRepository:
    """
    Содержит информацию о репозитории на GitHub.

    URL репозитория хранится в формате отельных компонентов для удобного доступа к его частям.
    Так же инстанс класса наполняется данными от API по коммитам, задачам и пулл-реквестам.
    """

    scheme: str
    netloc: str
    owner: str
    repo: str
    branch: str

    commits: Optional[List[Commit]] = None
    pull_requests: Optional[List[PullRequest]] = None
    issues: Optional[List[Issue]] = None

    def __init__(self, url: str, branch: str):
        self.branch = branch

        url_components = urlparse(url)
        self.scheme = url_components.scheme
        assert self.scheme == "https", f"В указанном URL({url}) протокол отличается от https: {url_components.scheme}"
        self.netloc = url_components.netloc
        if self.netloc != "github.com":
            raise NotImplementedError(f"Реализована работа только с репозиториями на Github.com.")
        assert url_components.path, f"В указанном URL({url}) не найден путь к репозиторию."
        self.owner, self.repo = url_components.path.split("/")[1:]

    def __str__(self):
        return f"{self.scheme}://{self.netloc}/{self.owner}/{self.repo}"

    @staticmethod
    def paginated_search_request(path: str, params: Dict, per_page: int = 100) -> Tuple[List[Dict], bool]:
        """
        Получает все результаты по поисковому запросу к search API используя пагинацию.

        Дополнительно информирует с помощью incomplete_set_flag вызывающий код о том, что
        результаты не могут быть получены полностью и требуется дальнейшая сегментация
        целевого набора с помощью параметров поиска в новых запросах.

        :param path: Путь в search API. (issues, commits и т.п.)
        :param params: GET-параметры которые будут переданы API в итоговом URL запроса.
        :param per_page: Кол-во результатов на одной странице.
        :return: Десериализированный контент от API в виде словаря или листа словарей(в зависимости от запроса).
        """

        items = list()
        incomplete_set_flag = False

        pages = 1
        page = 1
        while page <= pages:
            params["per_page"] = per_page
            params["page"] = page
            try:
                data = github_api_request(path=f"/search/{path}", params=params)
            except SearchResultsLimitReached:
                incomplete_set_flag = True
                break

            items += data["items"]
            pages = int((decimal.Decimal(data['total_count']) / per_page).quantize(1, decimal.ROUND_UP))
            page += 1

        return items, incomplete_set_flag

    def search_objects(self, object_class: Type[GithubObject], start_date: date, end_date: date, search_query: str) -> Set[Any]:
        """
        Проводит поиск объектов используя search API GitHub'а.
        В поисковом запросе использует даты создания объектов для получения релевантных результатов и
        для преодоления ограничения на 1000 результатов по запросу в случае больших диапазонов дат.

        :param object_class: Ссылка на класс объектов, по которым будет вестись поиск.
        :param start_date: Начальная дата диапазона.
        :param end_date: Конечная дата диапазона.
        :param search_query: Поисковой запрос(дополняется диапазоном дат в ходе работы)
        :return: Сет инстансов объектов запрошенного класса.
        """

        objects = set()

        incomplete_set_flag = True
        while incomplete_set_flag:
            items, incomplete_set_flag = self.paginated_search_request(
                path=object_class.api_path,
                params=dict(
                    sort="created",
                    order="asc",
                    q=f"{search_query}+created:{start_date.strftime('%Y-%m-%d')}..{end_date.strftime('%Y-%m-%d')}"
                )
            )

            for item in items:
                obj = object_class.from_dict(data=item, base=self.branch)
                objects.add(obj)

            # Начальная дата диапазона заменяется на дату создания самого
            # свежего из объектов которые уже удалось получить.
            if incomplete_set_flag:
                start_date = obj.created_at
            else:
                break

        return objects

    def get_commits(self, start_date: date, end_date: date) -> List[Commit]:
        """
        Возвращает все коммиты на ветке репозитория в указанном временном диапазоне.

        Не использует search API т.к. search API индексирует коммиты только на ветке master.
        Вместо него используется обычный эндпоинт commits.

        :param start_date: Начальная дата диапазона.
        :param end_date: Конечная дата диапазона.
        """

        commits = list()

        per_page = 100
        page = 1
        while page:
            params = dict(
                sha=self.branch,
                per_page=per_page,
                page=page,
                since=start_date,
                until=end_date
            )

            data = github_api_request(path=f"/repos/{self.owner}/{self.repo}/commits", params=params)
            for element in data:
                commits.append(Commit.from_dict(element))

            if len(data) == per_page:
                page += 1
            else:
                page = None

        return commits

    def get_top_contributors(self, count: int = 30) -> List[Contributor]:
        """
        Исходя из имеющихся данных о коммитах(GithubRepository.commits) возвращает топ контрибьюторов.
        Метод не позиционирует в топе сумму анонимных коммитов(от контрибьюторов без логина на github.com).

        :param count: Желаемое количество самых активных контрибьюторов репозитория.
        """

        contributions = defaultdict(lambda: 0)
        for commit in self.commits:
            contributions[commit.contributor] += 1

        return sorted(
            [Contributor(node_id=k.node_id, login=k.login, contributions=v) for k, v in contributions.items() if k is not None],
            key=lambda x: x.contributions,
            reverse=True
        )[:count]

    def get_pull_requests(self, start_date: date, end_date: date) -> [PullRequest]:
        """Получает все пулл-реквесты, отфильтрованные по базовой ветке и диапазону дат с помощью search API."""

        pull_requests = self.search_objects(
            object_class=PullRequest,
            start_date=start_date,
            end_date=end_date,
            search_query=f"repo:{self.owner}/{self.repo}+is:pr+base:{self.branch}"
        )
        return list(pull_requests)

    def get_issues(self, start_date: date, end_date: date) -> List[Issue]:
        """Получает все задачи репозитория в диапазоне start_date-end_date, включительно."""

        issues = self.search_objects(
            object_class=Issue,
            start_date=start_date,
            end_date=end_date,
            search_query=f"repo:{self.owner}/{self.repo}+is:issue"
        )
        return list(issues)

    def pull_data(self, start_date: date, end_date: date) -> None:
        """Загрузка всех требуемых данных о репозитории для отчета."""

        self.pull_requests = self.get_pull_requests(start_date=start_date, end_date=end_date)
        self.issues = self.get_issues(start_date=start_date, end_date=end_date)
        self.commits = self.get_commits(start_date=start_date, end_date=end_date)


if __name__ == "__main__":
    logging.basicConfig(
        format='%(asctime)s[%(levelname)s]: %(message)s',
        handlers=[logging.StreamHandler(sys.stdout)],
        level=logging.DEBUG
    )

    args = parse_args()
    assert args.start_date <= args.end_date, f"Указанная дата начала анализа {args.start_date} больше, " \
        f"чем дата его завершения {args.end_date}."

    repository = GithubRepository(url=args.url, branch=args.branch)

    logging.debug(f"Начато получение информации о репозитории {repository}")
    repository.pull_data(start_date=args.start_date, end_date=args.end_date)
    logging.debug(f"Завершено получение информации о репозитории {repository}")
