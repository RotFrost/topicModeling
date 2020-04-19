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
                

        f = furl(response.url)
        f.args['page'] = int(f.args['page']) + 1
        next_page = f.url
        print(next_page)
        
        currentPage = int(f.args['page'])
        endPageNumber = max(map(int, response.css("li.bpa-pager-page-number *::text").getall()))     
        
        if currentPage <= endPageNumber:
            yield SplashRequest(next_page, callback=self.parse, args={'wait': 8}, dont_filter=True)
        
    
    def parse_press(self, response):
        def xpath_get(query):
            result = response.xpath(query).get()
            return "" if result is None else result.strip()

        
        def query_getall(query, css):
            if css == True:
                result = unicodedata.normalize("NFKD", ''.join(response.css(query).getall()))
                return "" if result is None else result.strip()
            else:
                result = unicodedata.normalize("NFKD", ''.join(response.xpath(query).getall()))
                return "" if result is None else result.strip()
        
        yield {
            'title': xpath_get('//span[contains(@class, "bpa-teaser-title-text-inner")]/text()'),
            'shortText': query_getall('//div[contains(@class, "bpa-short-text")]/p/text()', False),
            'pressMessage': xpath_get('//li[contains(@class, "bpa-collection-item")]/text()'),
            'time': xpath_get('//span[contains(@class, "bpa-time")]/time/text()'),
            'author': xpath_get('//span[contains(@class, "bpa-dash")]/text()'),
            'richText': query_getall('div.bpa-richtext *::text', True),
            'url': response.url
            }
 
#        def xpath_get(response, _xpath):
#            return response.xpath(_xpath).get().strip()
#
#        next_page = response.css('li.next a::attr(href)').get()
#       if next_page is not None:
#           yield response.follow(next_page, self.parse)
            
#docker pull scrapinghub/splash
#docker run -p 8050:8050 scrapinghub/splash
#scrapy crawl quotes -o press.json

'''         
https://www.bundesregierung.de/breg-de/aktuelles/pressemitteilungen/buerokratieabbau-rechtsbereinigung-hebt-ueber-1-000-vorschriften-auf-758906

'title':response.xpath('//span[contains(@class, "bpa-teaser-title-text-inner")]/text()').get(),
'shortText': response.xpath('//div[contains(@class, "bpa-short-text")]/p/text()').getall(),
'pressMessage': response.xpath('//li[contains(@class, "bpa-collection-item")]/text()').get(),
'time': response.xpath('//span[contains(@class, "bpa-time")]/time/text()').get(),
'author': response.xpath('//span[contains(@class, "bpa-dash")]/text()').get(),
'richText': response.css('div.bpa-richtext *::text').getall()

    def parse(self, response):
        for _url in response.css("div.bpa-teaser-text-wrapper a::attr(href)").extract():
            yield {
                'link': _url
            }
                   
            
                        'author': response.xpath('//span[contains(@class, "bpa-dash")]/text()').get().encode('utf-8'),
                        
                        
                                if self.count == 3:
            return None;
        
        ++self.count
        next_page = self.baseUrl+self.count
        if next_page is not None:
            next_page = response.urljoin(next_page)
            yield scrapy.Request(next_page, callback=self.parse)
    
            yield {
            'title':self.xpath_get(reponse,'//span[contains(@class, "bpa-teaser-title-text-inner")]/text()'),
            'shortText': self.list_to_string(response.xpath('//div[contains(@class, "bpa-short-text")]/p/text()').getall()),
            'pressMessage': self.xpath_get(reponse,'//li[contains(@class, "bpa-collection-item")]/text()'),
            'time': self.xpath_get(reponse,'//span[contains(@class, "bpa-time")]/time/text()'),
            'author': self.xpath_get(reponse,'//span[contains(@class, "bpa-dash")]/text()'),
            'richText': self.list_to_string(response.css('div.bpa-richtext *::text').getall()),
            'url': response.url
            }
    
    def list_to_string(_list):
        return unicodedata.normalize("NFKD", ''.join(_list))
    
    def xpath_get(response, xpath):
        return response.xpath(xpath).get().strip()
    
    
              'title': xpath_get('//span[contains(@class, "bpa-teaser-title-text-inner")]/text()'),
            'shortText': query_getall('//div[contains(@class, "bpa-short-text")]/p/text()', False),
            'pressMessage': xpath_get('//li[contains(@class, "bpa-collection-item")]/text()'),
            'time': xpath_get('//span[contains(@class, "bpa-time")]/time/text()'),
            'author': xpath_get('//span[contains(@class, "bpa-dash")]/text()'),
            'richText': query_getall('div.bpa-richtext *::text', True),
            '''