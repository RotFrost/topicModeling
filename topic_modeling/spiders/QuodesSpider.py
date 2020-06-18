import scrapy
from scrapy_splash import SplashRequest
from furl import furl
import unicodedata

class QuotesSpider(scrapy.Spider):
    name = "quotes"
    
    def start_requests(self):
        start_url = 'https://www.bundesregierung.de/breg-de/aktuelles/pressemitteilungen/?page=0'
        yield SplashRequest(start_url, self.parse, args={'wait': 8}, dont_filter=True)

    def parse(self, response):
        press_links = response.css("div.bpa-teaser-text-wrapper a::attr(href)").extract()
        yield from response.follow_all(press_links, self.parse_press)
                
        #ermittelt die nächste Seite mittels des query parameter
        f = furl(response.url)
        f.args['page'] = int(f.args['page']) + 1
        next_page = f.url
        print(next_page)
        
        #Ermittelt die jetztige Seite mittels des query parameter
        currentPage = int(f.args['page'])
        #Ermittelt die letzte Seite
        endPageNumber = max(map(int, response.css("li.bpa-pager-page-number *::text").getall()))     
        
        #Falls die aktuelle Seite nicht die Letzte ist.
        if currentPage <= endPageNumber:
            yield SplashRequest(next_page, callback=self.parse, args={'wait': 8}, dont_filter=True)
        
    
    def parse_press(self, response):
        #für xpath_get requests
        def xpath_get(query):
            result = response.xpath(query).get()
            return "" if result is None else result.strip()

        #Für Query_getall requests
        def query_getall(query, css):
            if css == True:
                result = unicodedata.normalize("NFKD", ''.join(response.css(query).getall()))
                return "" if result is None else result.strip()
            else:
                result = unicodedata.normalize("NFKD", ''.join(response.xpath(query).getall()))
                return "" if result is None else result.strip()
        
        #Erhebt die folgenden Daten indem die Funktion query_getall mit dem Query oder Xpath aufgrufen wird.
        yield {
            'title': xpath_get('//span[contains(@class, "bpa-teaser-title-text-inner")]/text()'),
            'shortText': query_getall('//div[contains(@class, "bpa-short-text")]/p/text()', False),
            'pressMessage': xpath_get('//li[contains(@class, "bpa-collection-item")]/text()'),
            'time': xpath_get('//span[contains(@class, "bpa-time")]/time/text()'),
            'author': xpath_get('//span[contains(@class, "bpa-dash")]/text()'),
            'richText': query_getall('div.bpa-richtext *::text', True),
            'url': response.url
            }
 
            
#docker pull scrapinghub/splash
#docker run -p 8050:8050 scrapinghub/splash
#scrapy crawl quotes -o press.json

