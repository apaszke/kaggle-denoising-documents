import PIL
from PIL import ImageFont
from PIL import Image
from PIL import ImageDraw
from random import randint

num_images = 10

fonts = [
    ImageFont.truetype(font="Avenir", size=31, index=10)
]

lines = """\
Lorem ipsum dolor sit amet, consectetur adipiscing elit.
Proin nibh augue, suscipit a, scelerisque sed, lacinia in,
mi. Cras vel lorem. Etiam pellentesque aliquet tellus.
Phasellus pharetra nulla ac diam. Quisque semper justo at
risus. Donec venenatis, turpis vel hendrerit interdum, dui
ligula ultricies purus, sed posuere libero dui id orci. Nam
congue, pede vitae dapibus aliquet, elit magna vulputate
arcu, vel tempus metus leo non est. Etiam sit amet lectus
quis est congue mollis. Phasellus congue lacus eget neque.
Phasellus ornare, ante vitae consectetuer consequat, purus
sapien ultricies dolor, et mollis pede metus eget nisi.
Praesent sodales velit quis augue. Cras suscipit, urna at
aliquam rhoncus, urna quam viverra nisi, in interdum massa
nibh nec erat.\
""".split('\n')

backgrounds = [Image.open('data/bg/{}.png'.format(i), 'r').convert("L") for i in range(1, 5+1)]

def draw_text(draw):
    for i, line in enumerate(lines):
        draw.text((10, i*40 + 10), line, 0, font=fonts[0])

for i in range(1,num_images+1):
    img = Image.new("L", (540,420), 255)
    draw = ImageDraw.Draw(img)

    draw_text(draw)

    img.save('data/validation/y/{}.png'.format(i))

    img.paste(backgrounds[randint(0, len(backgrounds) - 1)], (0,0))
    draw_text(draw)

    img.save('data/validation/x/{}.png'.format(i))
