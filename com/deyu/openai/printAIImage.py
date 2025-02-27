import openai
if __name__ == '__main__':
    # 设置 OpenAI API 密钥
    openai.api_key = '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'
    response = openai.Image.create(
      prompt="a white siamese cat",
      n=1,
      size="1024x1024"
    )
    image_url = response['data'][0]['url']
    print(image_url)