# -*- coding: utf-8 -*-
"""
Use Wikipedia as a corpus for gensim

Does it make sense to train with gensim on a larger corpus than
the Bundespressemitteilungen???


Code is from Andrew Fairless (rather from his homepage)

http://afairless.com/the-peanuts-project/topic-modeling/making-the-wikipedia-corpus/making-the-wikipedia-corpus-full-code/
"""

#! /usr/bin/env python3

# moved these from function 'modify_text' to global variables for speed
from html2text import html2text
from nltk.tokenize import ToktokTokenizer
from nltk.corpus import stopwords
from nltk.stem.lancaster import LancasterStemmer


def decompress_bz2_file(filename, decompressed_filename):
    '''
    Decompresses 'bz2' file and saves it to a new file
    taken from:
    https://stackoverflow.com/questions/16963352/decompress-bz2-files
    '''

    from bz2 import BZ2File as bz2_file

    with open(decompressed_filename, 'wb') as new_file, bz2_file(filename, 'rb') as file:
        for data in iter(lambda: file.read(100 * 1024), b''):
            new_file.write(data)


def read_text_file(text_filename, as_string=False):
    '''
    reads each line in a text file as a list item and returns list by default
    if 'as_string' is 'True', reads entire text file as a single string
    '''
    text_list = []

    try:
        with open(text_filename) as text:
            if as_string:
                # reads text file as single string
                text_list = text.read().replace('\n', '')
            else:
                # reads each line of text file as item in a list
                for line in text:
                    text_list.append(line.rstrip('\n'))
            text.close()
        return(text_list)

    except:
        return('There was an error while trying to read the file')


def write_list_to_text_file(a_list, text_file_name, overwrite_or_append='a'):
    '''
    writes a list of strings to a text file
    appends by default; change to overwriting by setting to 'w' instead of 'a'
    '''

    try:
        textfile = open(text_file_name, overwrite_or_append, encoding='utf-8')
        for element in a_list:
            textfile.write(element)
            textfile.write('\n')

    finally:
        textfile.close()


def print_intermittent_status_message_in_loop(iteration, every_xth_iteration,
                                            total_iterations):
    '''
    Prints a message updating the user on the progress of a loop
    '''
    if iteration % every_xth_iteration == 0:
        import time
        print('Processing file {0} of {1}, which is {2:.0f}% at {3}'
            .format(iteration + 1, total_iterations,
                    100 * (iteration + 1) / total_iterations,
                    time.ctime(int(time.time())))
            )


def hms_string(sec_elapsed):
    '''
    # downloaded from:
    # http://www.heatonresearch.com/2017/03/03/python-basic-wikipedia-parsing.html
    # https://github.com/jeffheaton/article-code/blob/master/python/wikipedia/wiki-basic-stream.py
    # Simple example of streaming a Wikipedia
    # Copyright 2017 by Jeff Heaton, released under the The GNU Lesser General Public License (LGPL).
    # http://www.heatonresearch.com
    '''
    # Nicely formatted time string
    h = int(sec_elapsed / (60 * 60))
    m = int((sec_elapsed % (60 * 60)) / 60)
    s = sec_elapsed % 60
    return "{}:{:>02}:{:>05.2f}".format(h, m, s)


def strip_tag_name(t):
    '''
    # downloaded from:
    # http://www.heatonresearch.com/2017/03/03/python-basic-wikipedia-parsing.html
    # https://github.com/jeffheaton/article-code/blob/master/python/wikipedia/wiki-basic-stream.py
    # Simple example of streaming a Wikipedia
    # Copyright 2017 by Jeff Heaton, released under the The GNU Lesser General Public License (LGPL).
    # http://www.heatonresearch.com
    '''
    t = t.tag
    idx = t.rfind("}")
    if idx != -1:
        t = t[idx + 1:]
    return t


def process_wiki_xml(event, elem, tname, title, wiki_id, redirect, inrevision,
                    ns, page_text, iter_count, save_pages=False):
    '''
    Processes element extracted from Wikipedia XML file to allow classification
        of page into template, redirect, or article page and returns important
        information from page (e.g., page title, page Wikipedia ID number,
        article page text)

    # adapted from:
    # http://www.heatonresearch.com/2017/03/03/python-basic-wikipedia-parsing.html
    # https://github.com/jeffheaton/article-code/blob/master/python/wikipedia/wiki-basic-stream.py
    # Simple example of streaming a Wikipedia
    # Copyright 2017 by Jeff Heaton, released under the The GNU Lesser General Public License (LGPL).
    # http://www.heatonresearch.com
    # http://www.ibm.com/developerworks/xml/library/x-hiperfparse/
    '''

    if event == 'start':
        if tname == 'page':
            title = ''
            wiki_id = -1
            redirect = ''
            inrevision = False
            ns = 0
        elif tname == 'revision':
            inrevision = True

    else:
        if tname == 'title':
            if save_pages:
                title = elem.text                   # page title
            else:
                pass
        elif tname == 'id' and not inrevision:      # excludes IDs for pages being revised
            if save_pages:
                wiki_id = int(elem.text)            # Wikipedia page ID number
            else:
                pass
        elif tname == 'redirect':
            redirect = elem.attrib['title']
        elif tname == 'ns':
            ns = int(elem.text)
        elif tname == 'text':
            if save_pages:
                page_text = elem.text               # content from articles page
            else:
                pass

    return(ns, redirect, wiki_id, title, page_text)


def count_index_articles(ns, redirect, print_status_interval, num_documents,
                        sampling_interval, template_count, redirect_count,
                        article_count, sampled_article_count):
    '''
    Receives information from parsing through Wikipedia XML tree and counts
        the types of pages
    The 3 types of Wikipedia pages are:  template, redirect, article

    'sampled_article_count' - if only a portion of the article pages are being
        sampled (i.e., 'sampling_interval' > 1), this number will be lower than
        'article_count'

    # adapted from:
    # http://www.heatonresearch.com/2017/03/03/python-basic-wikipedia-parsing.html
    # https://github.com/jeffheaton/article-code/blob/master/python/wikipedia/wiki-basic-stream.py
    # Simple example of streaming a Wikipedia
    # Copyright 2017 by Jeff Heaton, released under the The GNU Lesser General Public License (LGPL).
    # http://www.heatonresearch.com
    # http://www.ibm.com/developerworks/xml/library/x-hiperfparse/
    '''
    article_end = False

    if ns == 10:
        template_count += 1

    elif len(redirect) > 0:
        redirect_count += 1

    else:
        article_count += 1
        print_intermittent_status_message_in_loop(
            article_count, print_status_interval, num_documents)
        if article_count % sampling_interval == 0:
            sampled_article_count += 1
            article_end = True

    return(template_count, redirect_count, article_count, sampled_article_count,
        article_end)


def modify_text(a_string):
    '''
    Processes Wikipedia text for analysis:  removes HTML tags, removes newline
        indicators, converts to lowercase, removes references, removes URLs,
        tokenizes, removes punctuations, removes stop words, removes numbers,
        stems
    Wikipedia text is input as a single string; each string is an article
    Returns a list of processed tokens

    Modifications have been made to the function to allow faster processing;
        original statements are retained as comments for portability and
        convenience when using the function in other situations
    '''

    #import html2text
    #import nltk
    # moved imports from 'html2text' and 'nltk' to global variables for speed
    #from html2text import html2text
    #from nltk.tokenize import ToktokTokenizer
    #from nltk.corpus import stopwords
    #from nltk.stem.lancaster import LancasterStemmer
    html_2_text = html2text
    nltk_tok_tok = ToktokTokenizer
    nltk_stopwords = stopwords
    nltk_lancaster = LancasterStemmer
    import string
    import re

    #nltk.download('punkt')
    #nltk.download('all')

    a_string = a_string.split('=References=')[0]                # remove references and everything afterwards
    #a_string = html2text.html2text(a_string).lower()            # remove HTML tags, convert to lowercase
    a_string = html_2_text(a_string).lower()                    # remove HTML tags, convert to lowercase
    a_string = re.sub(r'https?:\/\/.*?[\s]', '', a_string)      # remove URLs

    # 'word_tokenize' doesn't divide by '|' and '\n'
    # 'ToktokTokenizer' does divide by '|' and '\n', but retaining this
    #   statement seems to improve its speed a little
    a_string = a_string.replace('|', ' ').replace('\n', ' ')

    #tokens = nltk.tokenize.word_tokenize(a_string)
    #tokenizer = nltk.tokenize.ToktokTokenizer()                # tokenizes faster than 'word_tokenize'
    tokenizer = nltk_tok_tok()                                  # tokenizes faster than 'word_tokenize'
    tokens = tokenizer.tokenize(a_string)

    #stop_words = nltk.corpus.stopwords.words('english')
    stop_words = nltk_stopwords.words('english')
    string_punctuation = list(string.punctuation)
    remove_items_list = stop_words + string_punctuation
    tokens = [w for w in tokens if w not in remove_items_list]

    tokens = [w for w in tokens if '=' not in w]                        # remove remaining tags and the like
    tokens = [w for w in tokens if not                                  # remove tokens that are all digits or punctuation
            all(x.isdigit() or x in string_punctuation for x in w)]
    tokens = [w.strip(string.punctuation) for w in tokens]              # remove stray punctuation attached to words
    tokens = [w for w in tokens if len(w) > 1]                          # remove single characters
    tokens = [w for w in tokens if not any(x.isdigit() for x in w)]     # remove everything with a digit in it

    #stemmer = nltk.stem.PorterStemmer()
    #stemmer = nltk.stem.SnowballStemmer('english')
    #stemmer = nltk.stem.lancaster.LancasterStemmer()            # fastest stemmer; results seem okay
    stemmer = nltk_lancaster()                                  # fastest stemmer; results seem okay
    stemmed = [stemmer.stem(w) for w in tokens]

    return(stemmed)


def create_sqlite_database(database_name, template_table_name,
                        redirect_table_name, articles_table_name,
                        articles_text_col_name, key_col_name, processing_id):
    '''
    Creates sqlite database in which to store processed Wikipedia XML file
    Three tables are created in the database, one each for Wikipedia template
        pages, redirect pages, and article pages

    'database_name' - name of the database to create
    'template_table_name' - template pages table
    'redirect_table_name' - redirect pages table
    'articles_table_name' - articles pages table
    'articles_text_col_name' - column in articles table in which text of each
        article is stored
    'key_col_name' - column in articles table for primary key
    'processing_id" - column in articles table for index obtained when
        extracting the article from the Wikipedia XML dump file; used only for
        creating database and not a piece of information that is contained in
        Wikipedia itself
    '''
    import sqlite3

    con = sqlite3.connect(database_name)
    cur = con.cursor()

    cur.execute('CREATE TABLE {t} (wiki_id INTEGER, title TEXT)'
                .format(t=template_table_name))

    cur.execute('CREATE TABLE {t} (wiki_id INTEGER, title TEXT, redirect TEXT)'
                .format(t=redirect_table_name))

    cur.execute('CREATE TABLE {t} ({k} INTEGER PRIMARY KEY, wiki_id INTEGER, '
                'title TEXT, {te} TEXT, {pid} INTEGER)'
                .format(t=articles_table_name, k=key_col_name,
                        te=articles_text_col_name, pid=processing_id))

    con.commit()
    con.close()


def get_db_col_values(database_name, table_name, col_name, as_list=False):
    '''
    Retrieves values of a column from a SQLite database table
    If 'as_list' = True, returns values as items in a list
    If 'as_list' = False, returns values as a list of tuples where the values
        are the first items in each tuple
    '''
    import sqlite3

    con = sqlite3.connect(database_name)
    cur = con.cursor()

    cur.execute('SELECT {pid} FROM {at}'.format(pid=col_name, at=table_name))

    values = cur.fetchall()
    if as_list:
        values = [e[0] for e in values]

    con.commit()
    con.close()

    return(values)


def insert_row_sqlite(database_name, table_name, values_list):
    '''
    Inserts row into table of SQLite database
    'database_name' - name of SQLite database
    'table_name' - name of table to insert row into
    'values_list' - list of the row's values to insert
    WARNING:  do not use this function with an unsecured database; it is
        vulnerable to SQL injection attacks
    '''
    import sqlite3

    con = sqlite3.connect(database_name)
    cur = con.cursor()

    placeholders = ', '.join('?' * len(values_list))
    cur.execute('INSERT INTO {t} VALUES ({p})'
                .format(t=table_name, p=placeholders),
                (values_list))

    con.commit()
    con.close()


def insert_rows_sqlite(database_name, table_name, values_list):
    '''
    Inserts row into table of SQLite database
    'database_name' - name of SQLite database
    'table_name' - name of table to insert row into
    'values_list' - list of the row's values to insert
    WARNING:  do not use this function with an unsecured database; it is
        vulnerable to SQL injection attacks
    '''
    import sqlite3

    con = sqlite3.connect(database_name)
    cur = con.cursor()

    placeholders = ', '.join('?' * len(values_list[0]))
    #placeholders = [', '.join('?' * len(e)) for e in values_list]
    cur.executemany('INSERT INTO {t} VALUES ({p})'
                    .format(t=table_name, p=placeholders),
                    (values_list))

    con.commit()
    con.close()


def wiki_id_check(article_count, wiki_id):
    '''
    Checks that 'wiki_id' is a valid integer; if not, returns error message
    '''
    error_entry = None

    if not wiki_id:
        wiki_id = -1
        error_message = 'missing wiki_id'
        error_entry = [error_message, article_count, wiki_id]

    elif not isinstance(wiki_id, int):
        try:
            wiki_id = int(wiki_id)
            error_message = 'wiki_id not an integer, conversion successful'
            error_entry = [error_message, article_count, wiki_id]
        except:
            error_message = 'wiki_id not an integer, conversion failed'
            error_entry = [error_message, article_count, wiki_id]

    return(wiki_id, error_entry)


def title_check(article_count, wiki_id, title):
    '''
    Checks that 'title' is a valid string; if not, returns error message
    '''
    error_entry = None

    if not title:
        title = ''
        error_message = 'missing article title'
        error_entry = [error_message, article_count, wiki_id]

    elif not isinstance(title, str):
        try:
            title = str(title)
            error_message = 'title not a string, conversion successful'
            error_entry = [error_message, article_count, wiki_id, title]
        except:
            error_message = 'title not a string, conversion failed'
            error_entry = [error_message, article_count, wiki_id, title]

    return(title, error_entry)


def save_wiki_to_sql(ns, redirect, print_status_interval, num_documents, rows,
                    key_list, database_name, template_table_name,
                    redirect_table_name, articles_table_name, template_count,
                    redirect_count, article_count, sampled_article_count,
                    wiki_id, title, page_text, iter_count):
    '''
    # adapted from:
    # http://www.heatonresearch.com/2017/03/03/python-basic-wikipedia-parsing.html
    # https://github.com/jeffheaton/article-code/blob/master/python/wikipedia/wiki-basic-stream.py
    # Simple example of streaming a Wikipedia
    # Copyright 2017 by Jeff Heaton, released under the The GNU Lesser General Public License (LGPL).
    # http://www.heatonresearch.com
    # http://www.ibm.com/developerworks/xml/library/x-hiperfparse/
    '''

    # number of rows to insert into database at a time
    n_rows_to_save = 1000

    error_log_filename = 'error_log_stream.txt'

    if ns == 10:
        template_count += 1
        # commented out to save time
        #insert_row_sqlite(database_name, template_table_name,
        #                [wiki_id, title])

    elif len(redirect) > 0:
        redirect_count += 1
        # commented out to save time
        #insert_row_sqlite(database_name, redirect_table_name,
        #                [wiki_id, title, redirect])

    else:
        article_count += 1
        print_intermittent_status_message_in_loop(
            sampled_article_count, print_status_interval, num_documents)
        sampled_article_count += 1

        if page_text:
            try:
                page_text = modify_text(page_text)
                page_text = ' '.join(page_text)
            except:
                error_entry = ['error in modify_text function', iter_count,
                            article_count, wiki_id]
                write_list_to_text_file([str(error_entry)], error_log_filename)
                page_text = ''
        else:
            error_entry = ['missing article text', iter_count,
                        article_count, wiki_id]
            write_list_to_text_file([str(error_entry)], error_log_filename)
            page_text = ''

        #wiki_id, error_entry = wiki_id_check(article_count, wiki_id)
        #if error_entry:
        #    write_list_to_text_file([str(error_entry)], error_log_filename)

        #title, error_entry = title_check(article_count, wiki_id, title)
        #if error_entry:
        #    write_list_to_text_file([str(error_entry)], error_log_filename)

        row_values = (key_list[sampled_article_count-1],
                    wiki_id, title, page_text, iter_count)
        rows.append(row_values)
        if len(rows) >= n_rows_to_save or sampled_article_count == num_documents:
            try:
                insert_rows_sqlite(database_name, articles_table_name, rows)
                rows = []
            except:
                error_items = [e[4] for e in rows]
                error_entry = ['database insert failed', error_items]
                write_list_to_text_file([str(error_entry)], error_log_filename)
                rows = []
        #values_list = [key_list[sampled_article_count-1],
        #               wiki_id, title, page_text]
        #insert_row_sqlite(database_name, articles_table_name, values_list)
    return(template_count, redirect_count, article_count, sampled_article_count,
        rows)


def process_wiki(wiki_path, print_status_interval, num_documents, article_idx,
                database_names, key_list, sampling_interval=1,
                save_pages=False):
    '''
    Processes a dumped Wikipedia XML file and stores the results in a SQLite
        database

    'wiki_path' - file path to the Wikipedia XML file
    'database_names' - names for database to create, template table,
        redirect table, articles table, articles table's text column, and
        article table's primary key column
    'key_list' - list of integers to become the primary key for the table of
        articles in the database
    'print_status_interval' - the number of articles to process before providing
        a status update message (e.g., print the message every 50 articles)
    'num_documents' - the number of pages with articles in the Wikipedia XML
        file
    'sampling_interval' - process only every Xth article page, where X is the
        sampling interval

    Wikipedia periodically provides its entire website in a 'dump': further
        information here:
            https://meta.wikimedia.org/wiki/Data_dumps
            https://dumps.wikimedia.org/enwiki/
    This function divides the Wikipedia pages into template pages, redirect
        pages, and pages with articles; information on each is stored in its own
        table in the database
    Information on a template page includes its Wikipedia ID number and title
    Information on a redirect page includes its Wikipedia ID number, title, and
        and the title of the page that the user is redirected to
    Information on an article page includes its Wikipedia ID number, title, and
        the text of the page
    The text of an article page is processed for analysis by a call to the
        function 'modify_text' before being stored in the database
    The function returns the name of the articles table, the name of the column
        with the processed article text, and the name of the column with the
        primary key

    # adapted from:
    # http://www.heatonresearch.com/2017/03/03/python-basic-wikipedia-parsing.html
    # https://github.com/jeffheaton/article-code/blob/master/python/wikipedia/wiki-basic-stream.py
    # Simple example of streaming a Wikipedia
    # Copyright 2017 by Jeff Heaton, released under the The GNU Lesser General Public License (LGPL).
    # http://www.heatonresearch.com
    # http://www.ibm.com/developerworks/xml/library/x-hiperfparse/
    '''
    import xml.etree.ElementTree as etree
    import time

    if save_pages:
        database_name = database_names[0]
        template_table_name = database_names[1]
        redirect_table_name = database_names[2]
        articles_table_name = database_names[3]

    iter_count = 0
    page_count = 0
    article_count = 0
    redirect_count = 0
    template_count = 0
    sampled_article_count = 0

    title = ''
    wiki_id = -1
    redirect = ''
    inrevision = False
    ns = 0
    page_text = ''

    if save_pages:
        num_documents = len(article_idx)
        sampled_article_count = 0
        rows = []

    start_time = time.time()

    for event, elem in etree.iterparse(wiki_path, events=('start', 'end')):

        tname = strip_tag_name(elem)

        ns, redirect, wiki_id, title, page_text = process_wiki_xml(
            event, elem, tname, title, wiki_id, redirect, inrevision, ns,
            page_text, iter_count, save_pages)

        if tname == 'page' and event != 'start':

            page_count += 1

            # error inspection / debugging
            #if article_count > 21999 and article_count < 22003:
            #    write_list_to_text_file([page_text], 'wiki_article' + str(article_count) + '.txt', 'w')

            # instead of passing so many variables to and from the same
            #   functions, probably better to create a class object so that it
            #   can be passed and its state persists across loop iterations
            if not save_pages:
                (template_count, redirect_count, article_count,
                sampled_article_count, article_end) = count_index_articles(
                    ns, redirect, print_status_interval, num_documents,
                    sampling_interval, template_count, redirect_count,
                    article_count, sampled_article_count)
                if article_end:
                    article_idx.append(iter_count)
            elif iter_count in article_idx:
            #elif iter_count in article_idx and article_count > 60000:
                (template_count, redirect_count, article_count,
                sampled_article_count, rows) = save_wiki_to_sql(
                    ns, redirect, print_status_interval, num_documents, rows,
                    key_list, database_name, template_table_name,
                    redirect_table_name, articles_table_name, template_count,
                    redirect_count, article_count, sampled_article_count,
                    wiki_id, title, page_text, iter_count)
                # save a few unmodified articles to text files for testing
                #if article_count < 12:
                #    write_list_to_text_file([page_text], 'wiki_article' + str(article_count) + '.txt', 'w')

            elem.clear()

        iter_count += 1

    elapsed_time = time.time() - start_time

    print('Number of template pages: {}'.format(template_count))
    print('Number of redirect pages: {}'.format(redirect_count))
    print('Number of article pages: {}'.format(article_count))
    print('Number of total pages: {}'.format(page_count))
    print('Elapsed time: {}'.format(hms_string(elapsed_time)))

    if not save_pages:
        return(article_idx)
    else:
        return()


def split_list(a_list, n_parts):
    '''
    Splits a list into (nearly) equal-sized parts and returns them as a list of
        lists
    taken from:
    https://stackoverflow.com/questions/2130016/splitting-a-list-of-into-n-parts-of-approximately-equal-length
    '''
    d, r = divmod(len(a_list), n_parts)
    return(list(a_list[i * d + min(i, r):(i + 1) * d + min(i + 1, r)]
                for i in range(n_parts)))


def pool_process_wiki(wiki_path, message_interval, article_idx, database_names,
                    key_list, sampling_interval, save_pages, n_jobs):
    '''
    Parallelizes processing of dumped Wikipedia XML file and storing of the
        results in a SQLite database

    'wiki_path' - file path to the Wikipedia XML file
    'message_interval' - the number of articles to process before providing
        a status update message (e.g., print the message every 50 articles)
    'article_idx' - positional indices of articles in the XML file to be
        processed
    'database_names' - names for database to create, template table,
        redirect table, articles table, articles table's text column, and
        article table's primary key column
    'key_list' - list of integers to become the primary key for the table of
        articles in the database
    'sampling_interval' - process only every Xth article page, where X is the
        sampling interval
    'save_pages' - if 'True', saves pages to database; if 'False', only counts
        the pages
    'n_jobs' - number of parallel jobs/processors to use
    '''
    from itertools import repeat
    from multiprocessing import Pool

    article_idx_lists = split_list(article_idx, n_jobs)
    key_lists = split_list(key_list, n_jobs)

    params = zip(repeat(wiki_path),
                repeat(message_interval),
                repeat(0),                 # dummy value for 'num_documents'
                article_idx_lists,
                repeat(database_names),
                key_lists,
                repeat(sampling_interval),
                repeat(save_pages))

    with Pool(processes=n_jobs) as pool:
        pool.starmap(process_wiki, params)


def iter_documents_sqlite(database_name, table_name, col_name, key_col_name):
    '''
    Iterates through database and returns each document (a list of tokens)
    '''
    import sqlite3

    con = sqlite3.connect(database_name)
    cur = con.cursor()

    cur.execute('SELECT COUNT(*) FROM {t}'.format(c=key_col_name, t=table_name))
    response = cur.fetchall()
    num_documents = response[0][0]
    print('Number of documents in database is ', num_documents)

    try:
        for i in range(num_documents):
            cur.execute('SELECT ({c1}) FROM {t} WHERE {c2}=?'
                        .format(c1=col_name, t=table_name, c2=key_col_name),
                        (i, ))
            response = cur.fetchall()
            if response:
                document = response[0][0]
            else:
                document = ''
            yield(document.split())

    finally:
        con.close()


class TheCorpusFromSql(object):
    '''
    Iterates through each document (a list of tokens) and creates a corpus and
        dictionary in accordance with the Gensim text analysis package
    '''

    def __init__(self, database_name, table_name, col_name, key_col_name):
        from gensim.corpora import Dictionary
        self.database_name = database_name
        self.table_name = table_name
        self.col_name = col_name
        self.key_col_name = key_col_name
        self.dictionary = Dictionary(iter_documents_sqlite(
            database_name, table_name, col_name, key_col_name))

    def __iter__(self):
        for document_tokens_list in iter_documents_sqlite(
            self.database_name, self.table_name, self.col_name, self.key_col_name):
            yield self.dictionary.doc2bow(document_tokens_list)


def main():
    '''
    Creates and saves corpus and dictionary from Wikipedia XML file dump

    Wikipedia periodically provides its entire website in a 'dump': further
        information here:
            https://meta.wikimedia.org/wiki/Data_dumps
            https://dumps.wikimedia.org/enwiki/


    Wikipedia XML dump download information:

    Main site version:
    2017-07-02 19:52:45 done Recombine articles, templates, media/file descriptions, and primary meta-pages.
        enwiki-20170701-pages-articles.xml.bz2 13.1 GB
    2017-07-02 16:48:31 done Articles, templates, media/file descriptions, and primary meta-pages.
        enwiki-20170701-pages-articles1.xml-p10p30302.bz2 156.8 MB


    Illinois mirror:
    ftp://ftpmirror.your.org/pub/wikimedia/dumps/enwiki/20170701/

    Illinois mirror version:
    File:enwiki-20170701-pages-articles.xml.bz2
        13784309 KB     7/2/17  7:52:00 PM
    File:enwiki-20170701-pages-articles1.xml-p10p30302.bz2
        160604 KB   7/2/17  3:46:00 PM


    Downloads, July 13, 2017:

    Entire Wikipedia dump file:
    ftp://ftpmirror.your.org/pub/wikimedia/dumps/enwiki/20170701/enwiki-20170701-pages-articles.xml.bz2

    Part of Wikipedia dump file, to use for smaller-scale testing:
    ftp://ftpmirror.your.org/pub/wikimedia/dumps/enwiki/20170701/enwiki-20170701-pages-articles1.xml-p10p30302.bz2

    '''

    import os
    import random
    import gensim
    import time

    start_time = time.time()


    # prepare, decompress Wikipedia XML dump file
    # -------------------------------------------------
    filepath = os.getcwd()
    #filename = 'enwiki-20170701-pages-articles1.xml-p10p30302.bz2'  # small part of Wikipedia
    filename = 'enwiki-20170701-pages-articles.xml.bz2'             # all of Wikipedia
    compressed_path = os.path.join(filepath, filename)
    wiki_path = compressed_path.rsplit('.', 1)[0]
    #decompress_bz2_file(compressed_path, wiki_path)


    # count and index Wikipedia articles
    # -------------------------------------------------

    message_interval = 100000                           # print status update message every 'x' documents
    #est_num_docs = 15151                                # estimated number of documents, small part of Wikipedia
    est_num_docs = 9019957                              # estimated number of documents, all of Wikipedia
    sampling_interval = 3
    article_idx_filename = 'article_idx.txt'
    article_idx_path = os.path.join(filepath, article_idx_filename)

    # if articles have already been counted and indexed, retrieve list
    if os.path.exists(article_idx_path):
        article_idx_str = read_text_file(article_idx_path)
        num_documents_all = len(article_idx_str)
        article_idx_str = article_idx_str[0::sampling_interval]
        article_idx = [int(e) for e in article_idx_str]

    # otherwise, count and index Wikipedia articles in Wikipedia XML dump file
    else:
        article_idx = []
        article_idx = process_wiki(wiki_path, message_interval, est_num_docs,
                                article_idx, '', [], sampling_interval)
        article_idx_str = [str(e) for e in article_idx]
        write_list_to_text_file(article_idx_str, article_idx_path, 'w')

    print('Wikipedia page indices loaded.')


    # count and index Wikipedia articles
    # -------------------------------------------------

    # database names:  names for database, template table, redirect table,
    #   articles table, articles table's text column, article table's
    #   primary key column, and processing ID for each article
    database_names = ['wiki_token_docs.sqlite', 'template', 'redirect',
                    'articles', 'text', 'key', 'processing_id']
    database_path = os.path.join(filepath, database_names[0])

    # if SQLite database doesn't already exist, create it
    if not os.path.exists(database_path):
        print('Creating database.')
        create_sqlite_database(database_names[0], database_names[1],
                            database_names[2], database_names[3],
                            database_names[4], database_names[5],
                            database_names[6])
        article_idx_done = []

    # or if SQLite database does already exist, retrieve indices of articles
    #   that have already been saved to it
    else:
        article_idx_done = get_db_col_values(
            database_names[0], database_names[3], database_names[6], True)

    article_idx_do = list(set(article_idx) - set(article_idx_done))

    print('{} articles are already saved to the database.'
        .format(len(article_idx_done)))
    print('{} processed articles will be saved to the database.'
        .format(len(article_idx_do)))

    if len(article_idx_do) == 0:
        print('All articles have been added to the database.')
        return


    # process, save Wikipedia to SQLite database
    # -------------------------------------------------

    # set up article indices and database table keys, excluding any already in
    #   the database
    num_documents_do = len(article_idx_do)
    key_list_all = range(num_documents_all)
    key_list_used = get_db_col_values(
        database_names[0], database_names[3], database_names[5], True)
    key_list_available = set(key_list_all) - set(key_list_used)

    # randomizes order of Wikipedia articles when accessing by table primary key
    random.seed(513598)
    key_list = random.sample(key_list_available, len(key_list_available))

    save_pages = True
    n_jobs = 4

    # print status update message every 'x' documents
    message_interval = max(((num_documents_do / n_jobs) // 1000), 1)

    if n_jobs == 1:
        process_wiki(wiki_path, message_interval, num_documents_do,
                    article_idx_do, database_names, key_list,
                    sampling_interval, save_pages=True)
    else:
        pool_process_wiki(wiki_path, message_interval, article_idx_do,
                        database_names, key_list, sampling_interval,
                        save_pages, n_jobs)


    # delete de-compressed Wikipedia dump file
    # -------------------------------------------------
    #os.remove(wiki_path)


    # create and save Gensim corpus and dictionary
    # -------------------------------------------------
    wiki_corpus = TheCorpusFromSql(database_names[0], database_names[3],
                                database_names[4], database_names[5])
    #wiki_dictionary = wiki_corpus.dictionary
    wiki_corpus.dictionary.save('wiki_dictionary.dict')
    wiki_corpus.dictionary.save_as_text('wiki_dictionary.txt')
    gensim.corpora.MmCorpus.serialize('wiki_corpus.mm', wiki_corpus)


    # -------------------------------------------------
    elapsed_time = time.time() - start_time
    print('Finished at {ct}'.format(ct=time.ctime(int(time.time()))))
    print('Elapsed time: {}'.format(hms_string(elapsed_time)))


if __name__ == '__main__':
    main()