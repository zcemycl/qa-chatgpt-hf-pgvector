import guardrails as gd
from pydantic import BaseModel, Field


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
