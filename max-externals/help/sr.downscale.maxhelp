{
	"patcher": {
		"fileversion": 1,
		"appversion": {
			"major": 8,
			"minor": 6,
			"revision": 0,
			"architecture": "x64",
			"modernui": 1
		},
		"classnamespace": "box",
		"rect": [
			100.0,
			100.0,
			800.0,
			550.0
		],
		"description": "Pixelation/downscale effect",
		"digest": "Creates retro pixelated look by reducing effective resolution",
		"tags": "jitter, GPU, pixelate, downscale, retro, effect",
		"boxes": [
			{
				"box": {
					"id": "obj-1",
					"maxclass": "comment",
					"numinlets": 1,
					"numoutlets": 0,
					"patching_rect": [
						30.0,
						20.0,
						400.0,
						20.0
					],
					"text": "sr.downscale - Pixelation Effect",
					"fontsize": 14.0,
					"fontface": 1
				}
			},
			{
				"box": {
					"id": "obj-2",
					"maxclass": "comment",
					"numinlets": 1,
					"numoutlets": 0,
					"patching_rect": [
						30.0,
						45.0,
						600.0,
						40.0
					],
					"text": "Reduces effective resolution to create pixelated/retro look.\nscale 0 = no effect, scale 0.5 = 16px blocks, scale 1 = 256px blocks"
				}
			},
			{
				"box": {
					"id": "obj-world",
					"maxclass": "newobj",
					"numinlets": 1,
					"numoutlets": 0,
					"patching_rect": [
						550.0,
						100.0,
						180.0,
						22.0
					],
					"text": "jit.world sr_downscale_ctx @visible 0"
				}
			},
			{
				"box": {
					"id": "obj-3",
					"maxclass": "toggle",
					"numinlets": 1,
					"numoutlets": 1,
					"outlettype": [
						"int"
					],
					"patching_rect": [
						30.0,
						100.0,
						24.0,
						24.0
					]
				}
			},
			{
				"box": {
					"id": "obj-4",
					"maxclass": "newobj",
					"numinlets": 2,
					"numoutlets": 1,
					"outlettype": [
						"bang"
					],
					"patching_rect": [
						30.0,
						130.0,
						65.0,
						22.0
					],
					"text": "qmetro 30"
				}
			},
			{
				"box": {
					"id": "obj-5",
					"maxclass": "newobj",
					"numinlets": 1,
					"numoutlets": 2,
					"outlettype": [
						"jit_gl_texture",
						""
					],
					"patching_rect": [
						30.0,
						160.0,
						400.0,
						22.0
					],
					"text": "jit.movie @autostart 1 @loop 1 @output_texture 1 @drawto sr_downscale_ctx"
				}
			},
			{
				"box": {
					"id": "obj-20",
					"maxclass": "newobj",
					"numinlets": 1,
					"numoutlets": 1,
					"outlettype": [
						"bang"
					],
					"patching_rect": [
						400.0,
						100.0,
						58.0,
						22.0
					],
					"text": "loadbang"
				}
			},
			{
				"box": {
					"id": "obj-delay",
					"maxclass": "newobj",
					"numinlets": 2,
					"numoutlets": 1,
					"outlettype": [
						"bang"
					],
					"patching_rect": [
						250.0,
						130.0,
						63.0,
						22.0
					],
					"text": "delay 100"
				}
			},
			{
				"box": {
					"id": "obj-21",
					"maxclass": "message",
					"numinlets": 2,
					"numoutlets": 1,
					"outlettype": [
						""
					],
					"patching_rect": [
						250.0,
						160.0,
						120.0,
						22.0
					],
					"text": "read chickens.mp4"
				}
			},
			{
				"box": {
					"id": "obj-label-scale",
					"maxclass": "comment",
					"numinlets": 1,
					"numoutlets": 0,
					"patching_rect": [
						550.0,
						140.0,
						150.0,
						20.0
					],
					"text": "Scale: 0=none, 1=max (256px)"
				}
			},
			{
				"box": {
					"id": "obj-dial-scale",
					"maxclass": "dial",
					"numinlets": 1,
					"numoutlets": 1,
					"outlettype": [
						"float"
					],
					"patching_rect": [
						550.0,
						165.0,
						40.0,
						40.0
					],
					"size": 100.0,
					"min": 0.0,
					"mult": 0.01,
					"floatoutput": 1
				}
			},
			{
				"box": {
					"id": "obj-flonum-scale",
					"maxclass": "flonum",
					"numinlets": 1,
					"numoutlets": 2,
					"outlettype": [
						"",
						"bang"
					],
					"patching_rect": [
						600.0,
						175.0,
						60.0,
						22.0
					],
					"minimum": 0.0,
					"maximum": 1.0
				}
			},
			{
				"box": {
					"id": "obj-msg-scale",
					"maxclass": "message",
					"numinlets": 2,
					"numoutlets": 1,
					"outlettype": [
						""
					],
					"patching_rect": [
						600.0,
						205.0,
						80.0,
						22.0
					],
					"text": "scale $1"
				}
			},
			{
				"box": {
					"id": "obj-label-pixelate",
					"maxclass": "comment",
					"numinlets": 1,
					"numoutlets": 0,
					"patching_rect": [
						550.0,
						240.0,
						200.0,
						20.0
					],
					"text": "Pixelate: 0=pass-through, 1=pixelate"
				}
			},
			{
				"box": {
					"id": "obj-toggle-pixelate",
					"maxclass": "toggle",
					"numinlets": 1,
					"numoutlets": 1,
					"outlettype": [
						"int"
					],
					"patching_rect": [
						550.0,
						265.0,
						24.0,
						24.0
					]
				}
			},
			{
				"box": {
					"id": "obj-msg-pixelate",
					"maxclass": "message",
					"numinlets": 2,
					"numoutlets": 1,
					"outlettype": [
						""
					],
					"patching_rect": [
						600.0,
						265.0,
						90.0,
						22.0
					],
					"text": "pixelate $1"
				}
			},
			{
				"box": {
					"id": "obj-label-method",
					"maxclass": "comment",
					"numinlets": 1,
					"numoutlets": 0,
					"patching_rect": [
						550.0,
						300.0,
						200.0,
						20.0
					],
					"text": "Method: 0=nearest, 1=bilinear"
				}
			},
			{
				"box": {
					"id": "obj-toggle-method",
					"maxclass": "toggle",
					"numinlets": 1,
					"numoutlets": 1,
					"outlettype": [
						"int"
					],
					"patching_rect": [
						550.0,
						325.0,
						24.0,
						24.0
					]
				}
			},
			{
				"box": {
					"id": "obj-msg-method",
					"maxclass": "message",
					"numinlets": 2,
					"numoutlets": 1,
					"outlettype": [
						""
					],
					"patching_rect": [
						600.0,
						325.0,
						80.0,
						22.0
					],
					"text": "method $1"
				}
			},
			{
				"box": {
					"id": "obj-9",
					"maxclass": "newobj",
					"numinlets": 1,
					"numoutlets": 2,
					"outlettype": [
						"jit_gl_texture",
						""
					],
					"patching_rect": [
						30.0,
						210.0,
						480.0,
						22.0
					],
					"text": "jit.gl.pix sr_downscale_ctx @gen sr.downscale @scale 0.5 @pixelate 1 @method 0"
				}
			},
			{
				"box": {
					"id": "obj-10",
					"maxclass": "jit.pwindow",
					"numinlets": 1,
					"numoutlets": 2,
					"outlettype": [
						"jit_matrix",
						""
					],
					"patching_rect": [
						30.0,
						260.0,
						320.0,
						180.0
					]
				}
			},
			{
				"box": {
					"id": "obj-info",
					"maxclass": "comment",
					"numinlets": 1,
					"numoutlets": 0,
					"patching_rect": [
						550.0,
						370.0,
						200.0,
						60.0
					],
					"text": "Pixelate ON: creates blocky pixels\nPixelate OFF: pass-through (no effect)\nNearest: sharp edges\nBilinear: smooth interpolation"
				}
			}
		],
		"lines": [
			{
				"patchline": {
					"source": [
						"obj-20",
						0
					],
					"destination": [
						"obj-world",
						0
					]
				}
			},
			{
				"patchline": {
					"source": [
						"obj-20",
						0
					],
					"destination": [
						"obj-delay",
						0
					]
				}
			},
			{
				"patchline": {
					"source": [
						"obj-delay",
						0
					],
					"destination": [
						"obj-21",
						0
					]
				}
			},
			{
				"patchline": {
					"source": [
						"obj-21",
						0
					],
					"destination": [
						"obj-5",
						0
					]
				}
			},
			{
				"patchline": {
					"source": [
						"obj-3",
						0
					],
					"destination": [
						"obj-4",
						0
					]
				}
			},
			{
				"patchline": {
					"source": [
						"obj-4",
						0
					],
					"destination": [
						"obj-5",
						0
					]
				}
			},
			{
				"patchline": {
					"source": [
						"obj-5",
						0
					],
					"destination": [
						"obj-9",
						0
					]
				}
			},
			{
				"patchline": {
					"source": [
						"obj-dial-scale",
						0
					],
					"destination": [
						"obj-flonum-scale",
						0
					]
				}
			},
			{
				"patchline": {
					"source": [
						"obj-flonum-scale",
						0
					],
					"destination": [
						"obj-msg-scale",
						0
					]
				}
			},
			{
				"patchline": {
					"source": [
						"obj-msg-scale",
						0
					],
					"destination": [
						"obj-9",
						0
					]
				}
			},
			{
				"patchline": {
					"source": [
						"obj-toggle-pixelate",
						0
					],
					"destination": [
						"obj-msg-pixelate",
						0
					]
				}
			},
			{
				"patchline": {
					"source": [
						"obj-msg-pixelate",
						0
					],
					"destination": [
						"obj-9",
						0
					]
				}
			},
			{
				"patchline": {
					"source": [
						"obj-toggle-method",
						0
					],
					"destination": [
						"obj-msg-method",
						0
					]
				}
			},
			{
				"patchline": {
					"source": [
						"obj-msg-method",
						0
					],
					"destination": [
						"obj-9",
						0
					]
				}
			},
			{
				"patchline": {
					"source": [
						"obj-9",
						0
					],
					"destination": [
						"obj-10",
						0
					]
				}
			}
		]
	}
}
