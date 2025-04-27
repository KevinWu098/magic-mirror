
import { ProductBubble } from "@/components/product-bubble";

export default function ProductsPage() {
  // temp sample data
  const visual_matches = [
    {
      "position": 1,
      "title": "Soviet T-34/76 Shirt WW2 Modeling Tanks World War Panzer 131 USSR WOT | eBay",
      "link": "https://www.ebay.com/itm/275628577352",
      "source": "www.ebay.com",
      "source_icon": "https://encrypted-tbn2.gstatic.com/faviconV2?url=https://www.ebay.com&client=HORIZON&size=96&type=FAVICON&fallback_opts=TYPE,SIZE,URL&nfrp=2",
      "thumbnail": "https://encrypted-tbn1.gstatic.com/images?q=tbn:ANd9GcQG7Led1lVQ2OdWiUqtT52SDHknqSuGabvv7hQHaQOaXhu9CYy-",
      "thumbnail_width": 225,
      "thumbnail_height": 225,
      "image": "https://i.ebayimg.com/images/g/HAwAAOSwnvpjxrlG/s-l1200.jpg",
      "image_width": 1200,
      "image_height": 1200,
      "price": "$27*",
      "availability": "In stock"
    },
    {
      "position": 2,
      "title": "M60A2 Main Battle Tank Unisex Heavy Cotton Tee - Etsy",
      "link": "https://www.etsy.com/listing/1410993191/m60a2-main-battle-tank-unisex-heavy",
      "source": "www.etsy.com",
      "source_icon": "https://encrypted-tbn2.gstatic.com/faviconV2?url=https://www.etsy.com&client=HORIZON&size=96&type=FAVICON&fallback_opts=TYPE,SIZE,URL&nfrp=2",
      "thumbnail": "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcSjEGPyoPLPLeCF08BnTXeC44jX2DMiaI5pz474OXL06hRgQ296",
      "thumbnail_width": 225,
      "thumbnail_height": 225,
      "image": "https://i.etsystatic.com/13956121/r/il/773980/4645758363/il_fullxfull.4645758363_3xco.jpg",
      "image_width": 2048,
      "image_height": 2048,
      "price": "$15*",
      "availability": "In stock"
    },
    {
      "position": 3,
      "title": "Tank Man Men's Garment-dyed Heavyweight T-shirt - Liberty Maniacs",
      "link": "https://libertymaniacs.com/products/tank-man-men-s-garment-dyed-heavyweight-t-shirt",
      "source": "libertymaniacs.com",
      "source_icon": "https://encrypted-tbn2.gstatic.com/faviconV2?url=https://libertymaniacs.com&client=HORIZON&size=96&type=FAVICON&fallback_opts=TYPE,SIZE,URL&nfrp=2",
      "thumbnail": "https://encrypted-tbn2.gstatic.com/images?q=tbn:ANd9GcQa5LksYTRtdW8OMBr-GR-ADDLgLdhHKsgipEO2g_rI7BdPv4Ec",
      "thumbnail_width": 225,
      "thumbnail_height": 225,
      "image": "https://libertymaniacs.com/cdn/shop/products/mens-garment-dyed-heavyweight-t-shirt-grey-front-2-63b30a5dd1db1_2000x.jpg?v=1672678007",
      "image_width": 2000,
      "image_height": 2000,
      "price": "$35*",
      "availability": "In stock"
    }
  ];

  return (
    <main className="flex min-h-screen flex-col items-center p-8">
      <h1 className="mb-8 text-2xl font-bold">Similar Products</h1>
      
      <div className="flex flex-row gap-4 justify-center">
        {visual_matches.filter(e => e.price && e.availability === "In stock").slice(0, 3).map((product) => (
          <ProductBubble
            key={product.link}
            imageUrl={product.image} // or use thumbnail
            price={parseInt(product.price.replace(/[^0-9]/g, ''))}
            name={product.title}
            websiteUrl={product.link}
            websiteIcon={product.source_icon}
          />
        ))}
      </div>
    </main>
  );
} 