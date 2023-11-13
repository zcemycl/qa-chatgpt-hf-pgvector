from typing import Optional

import guardrails as gd
from pydantic import BaseModel, Field, HttpUrl


class ProductInfo(BaseModel):
    product_id: str = Field(description="product id")
    answer: str = Field(
        description="""Start the answer by praising the customer choice,
        Advice this fashion product,
        and Give cohesive product description."""
    )


prompt = """
As a shop assistant, given the following clothing information.
please extract a dictionary that answers ${question}.

product id: ${article_id}
product name: ${prod_name}
product type name: ${product_type_name}
product group name: ${product_group_name}
graphical appearance name: ${graphical_appearance_name}
color group name: ${colour_group_name}
garment group name: ${garment_group_name}
detailed description: ${detail_desc}

${gr.complete_json_suffix_v2}
"""


guard_product_advice = gd.Guard.from_pydantic(
    output_class=ProductInfo, prompt=prompt
)


class Message(BaseModel):
    url: Optional[HttpUrl] = Field(
        description="""Find the uri of the jpg or png or jpeg image.\
If the answer is not contained within the text below, say 'None'"""
    )
    path: Optional[str] = Field(
        description="""Find the path of the jpg or png or jpeg image.\
If the answer is not contained within the text below, say 'None'"""
    )
    request: str = Field(
        description="""Extract the client request for \
garments and remove image information. """
    )


prompt2 = """\
As a shop assistant, given the following information, ${question} \n
Please extract uri or path of the image file, and the client request.
"""

guard_image_search = gd.Guard.from_pydantic(
    output_class=Message, prompt=prompt2
)
