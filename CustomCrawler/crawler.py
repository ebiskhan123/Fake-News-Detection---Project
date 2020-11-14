import scrapy
import csv



class BrickSetSpider(scrapy.Spider) : 
    name = "brickset_spider"
    start_urls = ['http://brickset.com/sets/year-2016']

    col_headings ="name, pieces, minifigs, image\n"

    file =  open("lego.csv", "w") 
    file.write(col_headings)

    def parse(self, response) : 
        SET_SELECTOR = '.set'
        for brickset in response.css(SET_SELECTOR) : 
             
            NAME_SELECTOR = 'h1 a ::text'
            PIECE_SELECTOR = './/dl[dt/text() = "Pieces"]/dd/a/text()'
            MINIFIGS_SELECTOR = './/dl[dt/text() = "Minifigs"]/dd[2]/a/text()'
            IMAGE_SELECTOR = 'img ::attr(src)'

            name = brickset.css(NAME_SELECTOR).extract()[1][1:]
            pieces = brickset.xpath(PIECE_SELECTOR).extract_first()
            minifigs = brickset.xpath(MINIFIGS_SELECTOR).extract_first()
            image = brickset.css(IMAGE_SELECTOR).extract_first()

            yield {

                    'name' : name,
                    'pieces' : pieces,
                    'minifigs': minifigs,
                    'image': image
            }

            # item = name + ',' + pieces + ',' + minifigs + ',' + image  + '\n'         
            # # with open("lego.csv", "wb") as f : 
            # #     writer= csv.writer(f)
            # if item : 
            #     self.file.write(item)


            NEXT_PAGE_SELECTOR = '.next a ::attr(href)'
            next_page = response.css(NEXT_PAGE_SELECTOR).extract_first()
            if next_page : 
                yield scrapy.Request(
                        response.urljoin(next_page),
                        callback = self.parse
                    )
                
                