import scrapy


class TestSpider(scrapy.Spider):
    name = "test"
    allowed_domains = ["3rickdj.github.io"]
    start_urls = ["http://3rickdj.github.io/"]

    def parse(self, response):
        # Obtener el texto de todas las etiquetas <p>
        parrafos = response.css('p::text, b::text, a::text').getall()
        
        # Obtener enlaces
        enlaces = response.css('a::attr(href)').getall()
        
        # Contador para obtener solo n enlaces
        contador = 0
        
        # Obtener los primeros 3 enlaces
        for enlace in enlaces:
            # Seguir el enlace y llamar a la función parse_pagina
            yield response.follow(enlace, callback=self.parse_pagina)

        # Imprimir o guardar los párrafos según tus necesidades
        for parrafo in parrafos:
            print(parrafo)
            
    def parse_pagina(self, response):
        # Obtener el texto de todas las etiquetas <p>
        parrafos = response.css('p::text, b::text, a::text').getall()

        # Imprimir o guardar los párrafos según tus necesidades
        for parrafo in parrafos:
            print(parrafo)
